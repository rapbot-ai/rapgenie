#!/usr/bin/env python3
"""Downloads a W&B run's logged data to local files — history (every metric
logged over every step), config, and summary — instead of just printing a
sampled preview in a notebook cell.

Runs entirely on your own machine (not inside the training container) — it
only talks to W&B's public API, no GPU or RADTTS involved. Needs `wandb`,
`pandas`, and `pyyaml` installed locally (pip install wandb pandas pyyaml).

Usage (from the rapgenie repo root):
    python3 src/training/scripts/download_run_history.py <run_id> [--project <name>] [--entity <name>] [--output-dir <path>] [--sampled]
  or:
    npm run wandb:history -- <run_id>

Auth (WANDB_API_KEY) and --entity's default both come from the repo-root
.env, the same file every RunPod script already reads from — no separate
place to configure this script's credentials.

Always writes to <repo-root>/logs/weights-and-biases/<run_id>/ (history.csv +
history.meta.json) unless --output-dir overrides it — one fixed, predictable
place per run rather than wherever your cwd happened to be when you ran this.
(logs/ is gitignored — these are local exports, not checked-in artifacts.)

Full vs. sampled history: wandb's own run.history() silently caps out at
500 sampled points by default — fine for eyeballing a chart, wrong for a
"download the data" script, since a 10,000-iteration run would quietly come
back as 500 rows with no error. This script defaults to run.scan_history()
instead, which returns every logged point; pass --sampled if you actually
want the fast, capped preview.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_YAML = Path(__file__).resolve().parents[2] / "configs" / "train.yaml"

# Your own example command used this entity — kept as the built-in fallback
# so you don't have to retype it every run, but --entity or WANDB_ENTITY (in
# .env) always override it.
_DEFAULT_ENTITY = "rapbot-ai"


def _load_dotenv(path: Path) -> None:
    """Minimal, dependency-free .env loader — mirrors what every RunPod .js
    script's `require('dotenv').config({ path: ... })` already does for this
    repo: reads KEY=value lines from the repo-root .env and sets os.environ
    for any key not already exported in the shell (an existing env var
    always wins). No new pip dependency just for this one script."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        os.environ.setdefault(key, value)


def _default_project() -> str | None:
    """Reuses tracking.wandb.project from src/configs/train.yaml — the same
    single source of truth train.py's _init_wandb() already reads from —
    instead of hardcoding the project name a second time in this script,
    where it could silently drift out of sync."""
    if not TRAIN_YAML.exists():
        return None
    import yaml

    raw = yaml.safe_load(TRAIN_YAML.read_text()) or {}
    return ((raw.get("tracking") or {}).get("wandb") or {}).get("project")


def main() -> None:
    _load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Download a W&B run's history/config/summary to local files."
    )
    parser.add_argument("run_id", help="the run's id, e.g. lupefiasco-radtts-warmstart-v5-86f7089a5b26-77e3051")
    parser.add_argument(
        "--project",
        default=_default_project(),
        help=f"defaults to tracking.wandb.project in {TRAIN_YAML.relative_to(REPO_ROOT)}",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", _DEFAULT_ENTITY),
        help="defaults to $WANDB_ENTITY (.env) if set, else rapbot-ai",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="defaults to <repo-root>/logs/weights-and-biases/<run_id>/",
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="use wandb's fast, capped-at-500-points history() instead of the full, unsampled scan_history()",
    )
    args = parser.parse_args()

    if not args.project:
        print(
            f"Couldn't determine --project (no tracking.wandb.project in {TRAIN_YAML}). "
            "Pass --project explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    import wandb

    api = wandb.Api()
    run_path = f"{args.entity}/{args.project}/{args.run_id}"
    print(f"Fetching {run_path} ...")
    run = api.run(run_path)

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "logs" / "weights-and-biases" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.csv"
    if args.sampled:
        history_df = run.history()
    else:
        import pandas as pd

        history_df = pd.DataFrame(list(run.scan_history()))
    history_df.to_csv(history_path, index=False)
    print(f"Wrote {len(history_df)} rows -> {history_path}")

    meta_path = output_dir / "history.meta.json"
    meta = {
        "run_id": args.run_id,
        "name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "config": {k: v for k, v in run.config.items() if not k.startswith("_")},
        "summary": run.summary._json_dict,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote run config/summary -> {meta_path}")


if __name__ == "__main__":
    main()
