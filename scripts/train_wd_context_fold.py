"""Isolated process for one context-training fold; releases GPU memory on exit."""
import argparse
import json
from pathlib import Path
from llm.finetune_qwen3_8b_wd_text import main

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    c=json.loads(p.parse_args().config.read_text(encoding='utf-8'))
    for key in ['dataset','output']:c[key]=Path(c[key])
    main(argparse.Namespace(**c))
