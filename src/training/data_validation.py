"""Pre-flight validation of the aligned training dataset.

The notebook had no equivalent of this step: if a filelist referenced a missing
wav, or an alignment produced a zero-length clip, you'd find out partway
through an epoch, on a GPU you're paying for. This runs in seconds on CPU,
before a single GPU is allocated, as the first thing the training entrypoint
does.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


class DatasetValidationError(ValueError):
    pass


@dataclass
class FilelistIssue:
    line_number: int
    raw_line: str
    reason: str

    def __str__(self) -> str:
        return f"line {self.line_number}: {self.reason} ({self.raw_line!r})"


def parse_filelist_line(line: str) -> tuple[str, str]:
    """RADTTS filelists are pipe-delimited: `wavs/foo.wav|the transcript text`.
    Returns (relative_wav_path, transcript)."""
    parts = line.rstrip("\n").split("|")
    if len(parts) < 2:
        raise DatasetValidationError(f"expected 'path|text', got: {line!r}")
    return parts[0], parts[1]


def validate_filelist(
    filelist_path: Path,
    audio_dir: Path,
    min_duration_s: float = 0.1,
    max_duration_s: float = 10.2,
) -> list[FilelistIssue]:
    """Checks every row in a filelist: does the wav exist, is it a valid wav
    file, is its duration inside the configured [dur_min, dur_max] window
    (data_config.dur_min / dur_max in the old JSON config), and is the
    transcript non-empty. Returns all issues found rather than raising on the
    first one, so a single validation pass gives you the full picture.
    """
    issues: list[FilelistIssue] = []

    if not filelist_path.exists():
        raise DatasetValidationError(f"filelist not found: {filelist_path}")

    with filelist_path.open() as f:
        lines = [l for l in f.readlines() if l.strip()]

    if not lines:
        raise DatasetValidationError(f"filelist is empty: {filelist_path}")

    for i, line in enumerate(lines, start=1):
        try:
            rel_wav, transcript = parse_filelist_line(line)
        except DatasetValidationError as e:
            issues.append(FilelistIssue(i, line.strip(), str(e)))
            continue

        if not transcript.strip():
            issues.append(FilelistIssue(i, line.strip(), "empty transcript"))
            continue

        wav_path = audio_dir / rel_wav
        if not wav_path.exists():
            issues.append(FilelistIssue(i, line.strip(), f"missing audio file: {wav_path}"))
            continue

        try:
            with wave.open(str(wav_path), "rb") as wf:
                duration = wf.getnframes() / float(wf.getframerate())
        except Exception as e:
            issues.append(FilelistIssue(i, line.strip(), f"unreadable wav: {e}"))
            continue

        if not (min_duration_s <= duration <= max_duration_s):
            issues.append(
                FilelistIssue(
                    i,
                    line.strip(),
                    f"duration {duration:.2f}s outside [{min_duration_s}, {max_duration_s}]",
                )
            )

    return issues


def assert_dataset_ready(
    train_filelist: Path,
    val_filelist: Path,
    audio_dir: Path,
    min_duration_s: float = 0.1,
    max_duration_s: float = 10.2,
) -> None:
    """Entry point used by `training/train.py`, before launching the RADTTS
    subprocess. Raises with a readable summary if anything is wrong;
    otherwise returns silently."""
    all_issues: list[FilelistIssue] = []
    for name, fl in (("training", train_filelist), ("validation", val_filelist)):
        issues = validate_filelist(fl, audio_dir, min_duration_s, max_duration_s)
        if issues:
            summary = "\n  ".join(str(i) for i in issues[:20])
            more = f"\n  ...and {len(issues) - 20} more" if len(issues) > 20 else ""
            all_issues.append(f"{name} filelist ({fl}) has {len(issues)} issue(s):\n  {summary}{more}")

    if all_issues:
        raise DatasetValidationError("\n\n".join(all_issues))
