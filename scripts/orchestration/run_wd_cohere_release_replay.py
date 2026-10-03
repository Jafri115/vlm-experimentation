"""Prepare a local Cohere dataset release, then run the original ordinal suite."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(args):
    replacement = args.master_root.resolve()
    # Always check the supplied release; an old audit must not bypass a new release.
    if (replacement / 'replacement_audit.json').exists():
        raise ValueError('Replacement already exists. To resume training use '
                         'start_wd_cohere_ordinal_background.ps1 with -MasterRoot, '
                         'or choose a fresh -MasterRoot to prepare another release.')
    subprocess.run([sys.executable, '-u', str(REPO / 'scripts/data/prepare_wd_cohere_replacement.py'),
                    'from-directory', '--release-root', str(args.release_root.resolve()),
                    '--master-root', str(args.original_master.resolve()),
                    '--output', str(replacement)], cwd=REPO, check=True)
    subprocess.run([sys.executable, '-u', str(REPO / 'scripts/orchestration/run_wd_cohere_ordinal_replay.py'),
                    '--master-root', str(replacement), '--output', str(args.queue_root.resolve()),
                    '--run'], cwd=REPO, check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', type=Path, required=True)
    p.add_argument('--original-master', type=Path, required=True)
    p.add_argument('--master-root', type=Path, required=True)
    p.add_argument('--queue-root', type=Path, required=True)
    main(p.parse_args())
