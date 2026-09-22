"""Sequential presentation queue: baselines, content swaps, report, optional GPU seeds."""
import argparse
import datetime
import subprocess
import sys
from pathlib import Path

from analysis.wd_presentation_common import exclusive_lock,load_folds


def main(args):
    repo=Path(__file__).resolve().parents[1]
    master=args.master_root.resolve();root=args.output.resolve()
    load_folds(master)
    scripts=repo/'scripts'
    commands=[('tfidf',[sys.executable,str(scripts/'orchestration/run_wd_tfidf_baselines.py'),'--master-root',str(master),'--output',str(root/'baselines')]),
              ('content_swap',[sys.executable,str(scripts/'run_wd_content_swap.py'),'--master-root',str(master),
                '--llm-predictions',str(args.llm_root.resolve()/'oof_predictions.csv'),
                '--output',str(root/'content_swap'),'--shuffles','3'])]
    if args.include_vlm_swap:
        commands[1][1].extend(['--vlm-predictions',str(args.vlm_root.resolve()/'oof_predictions.csv')])
    report=[sys.executable,str(scripts/'report_wd_presentation.py'),'--master-root',str(master),
            '--root',str(root),'--output',str(root/'report'),'--specs',str(args.specs.resolve())]
    commands.append(('report',report))
    if args.run_seeds:
        commands.append(('controlled_seeds',[sys.executable,str(scripts/'repeat_wd_llm_seeds.py'),
            '--master-root',str(master),'--source-root',str(args.llm_root.resolve()),'--output',str(root/'seeds'),
            '--seeds','42','43','44']))
        commands.append(('report_with_seeds',report))
    import json
    specs=json.loads(args.specs.read_text(encoding='utf-8-sig'))
    required=[args.llm_root/'oof_predictions.csv']
    if args.include_vlm_swap:required.append(args.vlm_root/'oof_predictions.csv')
    required += [Path(s['path']) for s in specs if s['experiment'] in ['regression','consensus fine-tuning']]
    if args.run_seeds:
        required += [args.llm_root/f'fold_{f}'/'run_config.json' for f in range(1,6)]
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise SystemExit('Missing inputs:\n'+'\n'.join(missing))
    if args.dry_run:
        for name,command in commands:print(name,subprocess.list2cmdline(command))
        print('Preflight passed; no experiment executed.')
        return
    root.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(root/'queue.lock'):
        logs=root/'logs';logs.mkdir(exist_ok=True)
        def announce(message):
            line=f'[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
            print(line,flush=True)
            with (root/'queue.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
        announce('Presentation queue started')
        for name,command in commands:
            announce(f'START {name}')
            with (logs/f'{name}.log').open('a',encoding='utf-8') as log:
                log.write('\nCOMMAND: '+subprocess.list2cmdline(command)+'\n');log.flush()
                process=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:
                announce(f'FAILED {name}, exit {process.returncode}; inspect {logs/name}.log')
                raise SystemExit(process.returncode)
            announce(f'COMPLETED {name}')
        announce('Presentation queue finished')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-root',type=Path,default=Path('output/wd_multimodal_master_repaired'))
    p.add_argument('--output',type=Path,default=Path('output/wd_presentation_3day'))
    p.add_argument('--llm-root',type=Path,default=Path('output/llm_wd_soft_repaired_cv'))
    p.add_argument('--vlm-root',type=Path,default=Path('output/vlm_wd_soft_repaired_paired_cv'))
    p.add_argument('--specs',type=Path,default=Path(__file__).with_name('wd_pairwise_reliability_specs.json'))
    p.add_argument('--include-vlm-swap',action='store_true')
    p.add_argument('--run-seeds',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    main(p.parse_args())
