"""Build an offline subtitle editor and SRT exports for the WD transcript audit."""
import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_IDS = [
    '401001_S1_seg026', '402026_S6_seg017', '401006_S18_seg008',
    '401006_S18_seg009', '401033_S21_seg013', '401033_S21_seg014',
    '401033_S22_seg037', '401033_S22_seg038', '402004_S12_seg012',
    '402009_S4_seg023', '402009_S4_seg040', '402002_S5_seg028',
    '402002_S5_seg031', '401024_S14_seg002', '401024_S14_seg003',
    '401020_S5_seg043', '401016_S3_seg048', '401019_S2_seg038',
    '401001_S1_seg027', '402026_S6_seg033',
]


def stamp(value):
    ms = max(0, round(value * 1000))
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02},{ms:03}'


def parse_cues(text, start, end):
    pattern = re.compile(r'^\[(\d+):(\d+(?:\.\d+)?)\]\s*([TPU?]):\s*(.*)$')
    cues = []
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            minutes, seconds, role, words = match.groups()
            cues.append({'start': int(minutes)*60+float(seconds), 'role': role, 'text': words})
        elif line.strip():
            if not cues:
                raise ValueError(f'Unrecognized subtitle line: {line[:100]}')
            cues[-1]['text'] += '\n' + line.strip()
    for i, cue in enumerate(cues):
        later = [x['start'] for x in cues[i+1:] if x['start'] > cue['start']]
        inferred = min(later) if later else end
        cue['end'] = max(cue['start'] + .001, inferred)
        cue['end_estimated'] = True
    return cues


def srt(cues, origin=0, window=None):
    result = []
    for cue in cues:
        start, end = cue['start'], cue['end']
        if window:
            start, end = max(start, window[0]), min(end, window[1])
        if end <= start or end <= origin:
            continue
        result.append(f"{len(result)+1}\n{stamp(start-origin)} --> {stamp(end-origin)}\n{cue['role']}: {cue['text']}\n")
    return '\n'.join(result)


def build(args):
    with args.input.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    ids = args.ids or DEFAULT_IDS
    lookup = {r['sample_id']: r for r in rows}
    missing = set(ids)-lookup.keys()
    if missing:
        raise ValueError(f'Segment IDs not found: {sorted(missing)}')
    args.output.mkdir(parents=True, exist_ok=True)
    records, checks = [], []
    for sid in ids:
        r = lookup[sid]
        start, end = float(r['start_sec']), float(r['end_sec'])
        if end <= start:
            raise ValueError(f'{sid}: invalid segment interval')
        cues = parse_cues(r['transcript_text'], start, end)
        flags = []
        if any(c['start'] < start or c['start'] >= end for c in cues):
            flags.append('Transcript onset outside rating window; inspect boundary before trimming.')
        if any(cues[i]['start'] < cues[i-1]['start'] for i in range(1,len(cues))):
            flags.append('Non-monotonic transcript timestamps.')
        if len(set(c['start'] for c in cues)) != len(cues):
            flags.append('Simultaneous cue onsets; may reflect overlap or coarse timing.')
        record = {'id': sid, 'video': r['video'], 'video_path': r['video_path'],
                  'start': start, 'end': end, 'r1': r['WD_P_rater1'], 'r2': r['WD_P_rater2'],
                  'provider': r['transcript_provider'], 'flags': flags,
                  'source_flags': r.get('review_flags',''), 'raw': r['transcript_text'],
                  'annotation_start': r['segment_start'], 'annotation_end': r['segment_end'],
                  'cues': cues, 'notes': ''}
        records.append(record)
        # Session subtitles retain source onsets. Clip subtitles intersect the rating window.
        (args.output/f'{sid}.session.srt').write_text(srt(cues),encoding='utf-8-sig')
        (args.output/f'{sid}.clip.srt').write_text(srt(cues,start,(start,end)),encoding='utf-8-sig')
        checks.append({'sample_id': sid, 'start_sec':start,'end_sec':end,'cue_count':len(cues),
                       'first_onset':cues[0]['start'] if cues else '',
                       'last_onset':cues[-1]['start'] if cues else '',
                       'onsets_outside_window':sum(c['start']<start or c['start']>=end for c in cues),
                       'video_available_here':Path(r['video_path']).is_file(),
                       'review_flags':'; '.join(flags), 'source_flags':r.get('review_flags','')})
    with (args.output/'timing_checks.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    template = Path(__file__).with_name('wd_subtitle_review_template.html').read_text(encoding='utf-8')
    payload=json.dumps(records,ensure_ascii=False).replace('<','\\u003c')
    (args.output/'index.html').write_text(template.replace('__RECORDS_JSON__',payload),encoding='utf-8')
    (args.output/'README.md').write_text(
        '# WD subtitle review\n\nOpen index.html in a browser. Select a full-session video or an extracted clip. '
        'Select the matching media time mode. Full-session mode uses session timestamps; clip mode assumes '
        'the clip begins exactly at the rating-window start. Use the offset control if it begins elsewhere.\n\n'
        'Edits stay in memory until downloaded. Export corrections as JSON to preserve all cases and notes; '
        'import that JSON to resume. SRT files include speaker labels but do not encode human ratings.\n\n'
        'For the desktop Subtitle Edit application, open the video and corresponding .session.srt. '
        'Use .clip.srt only for a clip cut at the exact segment start. The HTML is a standalone reviewer, '
        'not the installed Subtitle Edit application.\n\n'
        'Cue ends are inferred from the next later onset (last cue: window end), not measured speech ends. '
        'Clip exports trim boundary-crossing cues and omit cues wholly outside the window. Session exports '
        'preserve source onsets. Original transcripts and ratings are unchanged. Numeric checks do not verify '
        'speaker identity, audible timing or the original coder label-window alignment.\n',encoding='utf-8')
    print(json.dumps({'html':str((args.output/'index.html').resolve()),'segments':len(records),
                      'srt_files':2*len(records),'segments_with_outside_onsets':sum(x['onsets_outside_window']>0 for x in checks)},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('output/wd_high_rater_segments/both_raters_3_or_higher.csv'))
    p.add_argument('--output',type=Path,default=Path('output/wd_high_rater_segments/subtitle_review'))
    p.add_argument('--ids',nargs='+',help='Optional segment IDs; default: 20 examples from the audit.')
    build(p.parse_args())
