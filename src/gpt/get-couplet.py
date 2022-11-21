import transformers
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import custom_fwd, custom_bwd
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from tqdm.auto import tqdm
import re
import random
import time
import uuid
import sys

class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
        self.bias = bias
 
    def forward(self, input):
        output = torch.clone(DequantizeAndLinear.apply(input, self.weight, self.absmax, self.code, self.bias))
        if self.adapter:
            output += self.adapter(input)
        return output
 
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"
 
 
class DequantizeAndLinear(torch.autograd.Function): 
    @staticmethod
    @custom_fwd
    def forward(ctx, input: torch.Tensor, weights_quantized: torch.ByteTensor,
                absmax: torch.FloatTensor, code: torch.FloatTensor, bias: torch.FloatTensor):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)
 
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert not ctx.needs_input_grad[1] and not ctx.needs_input_grad[2] and not ctx.needs_input_grad[3]
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias
 
 
class FrozenBNBEmbedding(nn.Module):
    def __init__(self, weight, absmax, code):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.adapter = None
 
    def forward(self, input, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(self.weight, absmax=self.absmax, code=self.code)
            output = F.embedding(input, weight_deq, **kwargs)
        if self.adapter:
            output += self.adapter(input)
        return output 
 
    @classmethod
    def from_embedding(cls, embedding: nn.Embedding) -> "FrozenBNBEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        return cls(weights_int8, *state)
 
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"
 
 
def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size: (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(input_chunk, code=code)
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)
 
    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)
 
 
def convert_to_int8(model):
    """Convert linear and embedding modules to 8-bit with optional adapters"""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr( 
                    module,
                    name,
                    FrozenBNBLinear(
                        weight=torch.zeros(child.out_features, child.in_features, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                        bias=child.bias,
                    ),
                )
            elif isinstance(child, nn.Embedding):
                setattr(
                    module,
                    name,
                    FrozenBNBEmbedding(
                        weight=torch.zeros(child.num_embeddings, child.embedding_dim, dtype=torch.uint8),
                        absmax=torch.zeros((child.weight.numel() - 1) // 4096 + 1),
                        code=torch.zeros(256),
                    )
                )

class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)

        convert_to_int8(self.attn)
        convert_to_int8(self.mlp)


class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)
        

class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_int8(self)


transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J

config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

start_time = time.time()
print('Loading GPT...')
gpt = torch.load("/home/ubuntu/rapgenie/src/models/gpt-j-8bit_002500.pt",  map_location=torch.device('cuda'))
end_time = time.time()
print('GPT loaded!')
print('Load time (seconds):', round(end_time - start_time, 2))
torch.cuda.empty_cache()

# GET TOPIC:

topic = sys.argv[1]
has_topic = len(topic) > 0
if has_topic:
  print("topic:", topic)

# GET RHYME SET

with torch.no_grad():
  prompt1 = "<" + topic + " =T2R="
  result_length = 30
  inputs = tokenizer(prompt1, return_tensors="pt").to('cuda:0')
  beam_outputs = gpt.generate(inputs["input_ids"],
    max_length=result_length,
    top_k=50, top_p=0.95, 
    do_sample=True, temperature=0.7, pad_token_id=50256,
    num_return_sequences=10)
  
  rhyme_sets = []
  topics = []

  for beam_output in beam_outputs:
    text = tokenizer.decode(beam_output, skip_special_tokens=True)
    rhyme_set = text[text.find(" =T2R= ")+len(" =T2R= "):text.rfind(">")]
    parts = rhyme_set.split(" \ ")
    rhymes = []
    for p in parts:
      rhyme = p.strip()
      if len(rhyme) > 0 and rhyme not in rhymes:
        rhymes.append(rhyme)
    if rhyme_set not in rhyme_sets and '=' not in rhyme_set:
      rhyme_sets.append(rhyme_set)
      if not has_topic:
        topics.append(rhymes[0])
torch.cuda.empty_cache()

# TODO: filter non-rhymes out of rhyme_sets using phonemizer

rhyme_sets.sort()

rhyme_text = random.sample(rhyme_sets, 1)[0]

if not has_topic:
  topic = random.sample(topics, 1)

print("rhyme-set:", rhyme_text)

# GET LINE 1:

with torch.no_grad():
  prompt2 = "<" + topic + ": " + rhyme_text + " =R2L= "
  result_length = 40
  inputs = tokenizer(prompt2, return_tensors="pt").to('cuda:0')
  beam_outputs = gpt.generate(inputs["input_ids"],
    max_length=result_length,
    top_k=50, top_p=0.95, 
    do_sample=True, temperature=0.7, pad_token_id=50256,
    num_return_sequences=10)

  first_rhyme = rhyme_text.split(" \ ")[0].strip().lower()
  first_lines = []

  for beam_output in beam_outputs:
    text = tokenizer.decode(beam_output, skip_special_tokens=True)
    text = ' '.join(text.split())
    trimmed_text = text[text.find(" =R2L= ")+len(" =R2L= "):text.rfind(">")]
    lines = trimmed_text.split(" / ")
    if len(lines) >=  1  and len(lines[0]) > 0:
      first_line = lines[0]
      last_word = first_line.split()[-1].strip().lower()
      last_word = re.sub(r'[^a-z]+', '', last_word)
      if first_rhyme == last_word and first_line not in first_lines:
        first_lines.append(first_line)
torch.cuda.empty_cache()

first_lines.sort()

print('first_lines:', first_lines)
first_line = random.sample(first_lines, 1)[0]

# GET LINE 2:

with torch.no_grad():
  prompt3 = "<" + topic + ": " + rhyme_text + " =R2L= " + first_line + " / "
  result_length = 80
  inputs = tokenizer(prompt3, return_tensors="pt").to('cuda:0')
  beam_outputs = gpt.generate(inputs["input_ids"],
    max_length=result_length,
    top_k=50, top_p=0.95, 
    do_sample=True, temperature=0.7, pad_token_id=50256,
    num_return_sequences=20)

  second_rhyme = rhyme_text.split(" \ ")[1].strip().lower()
  second_lines = []

  for beam_output in beam_outputs:
    text = tokenizer.decode(beam_output, skip_special_tokens=True)
    text = ' '.join(text.split())
    trimmed_text = text[text.find(" =R2L= ")+len(" =R2L= "):text.rfind(">")]
    lines = trimmed_text.split(" / ")
    if len(lines) >= 2 and len(lines[1]) > 0:
      second_line = lines[1]
      last_word = second_line.split()[-1].strip().lower()
      last_word = re.sub(r'[^a-z]+', '', last_word)
      if second_rhyme == last_word and second_line not in second_lines and '=' not in second_line:
        second_lines.append(second_line)

torch.cuda.empty_cache()

print('second_lines:', second_lines)
if len(second_lines) == 0:
  raise Exception("ERROR: second_lines.length is 0!")

second_lines.sort()

second_line = random.sample(second_lines, 1)[0]

print()
print('first_line:', first_line)
print("second_line:", second_line)

outputPath = sys.argv[2]

with open(outputPath, 'a') as file:
  file.write(first_line)
  file.write(' ')
  file.write(second_line)
