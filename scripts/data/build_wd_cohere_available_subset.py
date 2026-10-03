"""Build original-minute inputs from the provisional Memopsy Cohere word release.

Use source word timestamps plus the release's explicit word_updates overlay.
Never invent timestamps/roles. Missing sessions and empty windows are excluded.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def save_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n' for x in values), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_word(w):
    s, e = w.get('start'), w.get('end')
    return (isinstance(s, (float, int)) and isinstance(e, (float, int))
            and math.isfinite(s) and math.isfinite(e) and 0 <= s < e and bool(str(w.get('text','')).strip()))


def repair_unknown_roles(words, max_words=3, max_seconds=2.0, max_gap=1.0):
    """Fill short, timed UNKNOWN runs bracketed by the same original role.

    Never use newly inferred labels to infer further words. Raw diarization
    conflicts, untimed words and speaker changes are barriers.
    """
    output=[dict(w) for w in words]
    repairs=[]
    i=0
    while i<len(words):
        if words[i].get('speaker') in ('P','T'):
            i+=1
            continue
        start=i
        while i<len(words) and words[i].get('speaker') not in ('P','T'):
            i+=1
        end=i
        if start==0 or end==len(words) or end-start>max_words:
            continue
        before,after=words[start-1],words[end]
        role=before.get('speaker')
        if role not in ('P','T') or after.get('speaker')!=role:
            continue
        bracket=words[start-1:end+1]
        if not all(valid_word(w) for w in bracket):
            continue
        if any(b['index']!=a['index']+1 for a,b in zip(bracket,bracket[1:])):
            continue
        if words[end-1]['end']-words[start]['start']>max_seconds:
            continue
        if any(b['start'] < a['start'] or not -.05 <= b['start']-a['end'] <= max_gap
               for a,b in zip(bracket,bracket[1:])):
            continue
        known_raw={w.get('raw_speaker') for w in (before,after) if w.get('raw_speaker')}
        if len(known_raw)>1:
            continue
        if known_raw and any(w.get('raw_speaker') and w['raw_speaker'] not in known_raw
                             for w in words[start:end]):
            continue
        for position in range(start,end):
            output[position].update(speaker=role, dataset_role_inference='same_role_bracket_v1')
            repairs.append({'word_index':words[position]['index'],'word':words[position]['text'],
                            'start':words[position]['start'],'end':words[position]['end'],
                            'original_speaker':words[position].get('speaker'), 'inferred_speaker':role,
                            'left_word_index':before['index'],'right_word_index':after['index'],
                            'method':'same_role_bracket_v1','roles_verified':False})
    return output,repairs


ACKNOWLEDGMENTS={'ja','okay','ok','mhm','hm','hmm','aha','genau','nein','ah'}
PREFIXES={'und','aber','also','weil','dass','denn','wenn','obwohl','deshalb','deswegen'}


def terminal(text):
    return bool(re.search(r'[.!?][\"\u201c\u201d\u201e\u00ab\u00bb]*$',text.strip()))


def repair_contextual_roles(words):
    """Provisional, auditable punctuation/timing rules; not a German grammar model.

    Same-role islands may span up to 12 words/6 seconds. One-sided assignments
    need a sentence-tail or prefix cue, close timing and no raw-speaker conflict.
    Long gaps, overlap, untimed fragments and acknowledgments go to review.
    All anchors are original source labels; inferences never cascade.
    """
    output=[dict(w) for w in words]; repairs=[]; review=[]
    i=0
    while i<len(words):
        if words[i].get('speaker') in ('P','T'):
            i+=1; continue
        start=i
        # Do not let an unlabeled acknowledgment after a sentence get swallowed.
        while i<len(words) and words[i].get('speaker') not in ('P','T'):
            i+=1
            if terminal(str(words[i-1].get('text',''))): break
        end=i; fragment=words[start:end]
        left=start-1
        while left>=0 and words[left].get('speaker') not in ('P','T') and start-left<=12: left-=1
        right=end
        while right<len(words) and words[right].get('speaker') not in ('P','T') and right-end<12: right+=1
        before=words[left] if left>=0 and start-left<=12 and words[left].get('speaker') in ('P','T') else None
        after=words[right] if right<len(words) and right-end<12 and words[right].get('speaker') in ('P','T') else None
        record={'word_indices':[w['index'] for w in fragment],
                'text':' '.join(str(w.get('text','')) for w in fragment),
                'start':fragment[0].get('start'),'end':fragment[-1].get('end'),
                'left_role':before.get('speaker') if before else None,
                'right_role':after.get('speaker') if after else None}
        lexical=[re.sub(r'[^\w-]','',str(w.get('text','')).casefold()) for w in fragment]
        reason=None; role=None; method=None; support='moderate_heuristic'
        if not lexical or all(token in ACKNOWLEDGMENTS for token in lexical):
            reason='acknowledgment_requires_review'
        elif len(fragment)>12 or not all(valid_word(w) for w in fragment):
            reason='long_or_untimed_fragment'
        elif fragment[-1]['end']-fragment[0]['start']>6:
            reason='fragment_exceeds_six_seconds'
        elif any(b['index']!=a['index']+1 or not -.05<=b['start']-a['end']<=1.0
                 for a,b in zip(fragment,fragment[1:])):
            reason='discontinuous_or_overlapping_fragment'
        else:
            prev_gap=fragment[0]['start']-before['end'] if before and valid_word(before) else None
            next_gap=after['start']-fragment[-1]['end'] if after and valid_word(after) else None
            direct_prev=before is not None and left==start-1
            direct_next=after is not None and right==end
            if (direct_prev and direct_next and before['speaker']==after['speaker']
                    and prev_gap is not None and next_gap is not None
                    and -.05<=prev_gap<=1.5 and -.05<=next_gap<=1.5):
                role=before['speaker']; method='same_role_island_v2'; support='strong_context_rule'
            elif (direct_prev and prev_gap is not None and 0<=prev_gap<=.8
                    and not terminal(str(before.get('text','')))
                    and terminal(str(fragment[-1].get('text','')))):
                role=before['speaker']; method='previous_sentence_tail_v2'
            elif (direct_next and next_gap is not None and 0<=next_gap<=1.0
                    and lexical[0] in PREFIXES
                    and str(after.get('text',''))[:1].islower()):
                role=after['speaker']; method='next_sentence_prefix_v2'
            else:
                reason='insufficient_continuation_evidence'
            if role:
                anchor=before if method!='next_sentence_prefix_v2' else after
                anchor_raw=anchor.get('raw_speaker')
                if any(w.get('raw_speaker') and anchor_raw and w['raw_speaker']!=anchor_raw for w in fragment):
                    reason='raw_diarization_conflict'; role=None
                if role and method=='same_role_island_v2' and before.get('raw_speaker') and after.get('raw_speaker') and before['raw_speaker']!=after['raw_speaker']:
                    reason='raw_anchor_conflict'; role=None
                # No inferred word may overlap a nearby original contrary-role word.
                neighbours=words[max(0,start-12):min(len(words),end+12)]
                if role and any(w.get('speaker') in ('P','T') and w['speaker']!=role and valid_word(w)
                                and min(w['end'],fragment[-1]['end'])-max(w['start'],fragment[0]['start'])>.05
                                for w in neighbours):
                    reason='competing_labelled_speech_overlap'; role=None
        if not role:
            review.append({**record,'reason':reason}); continue
        for position in range(start,end):
            w=words[position]
            output[position].update(speaker=role,dataset_role_inference=method,
                                    role_source='context_inferred',role_confidence=support)
            repairs.append({'word_index':w['index'],'word':w['text'],'start':w['start'],'end':w['end'],
                            'original_speaker':w.get('speaker'),'inferred_speaker':role,'method':method,
                            'role_source':'context_inferred','confidence':support,'confidence_calibrated':False,
                            'left_word_index':before['index'] if before else None,
                            'right_word_index':after['index'] if after else None,'roles_verified':False})
    return output,repairs,review


def timestamp(seconds):
    tenths=int(round(seconds*10))
    minutes,remainder=divmod(tenths,600)
    return f'{minutes:02d}:{remainder//10:02d}.{remainder%10}'


def cue_groups(words):
    cues=[]
    for w in words:
        role = w.get('speaker') if w.get('speaker') in ('P','T') else 'UNKNOWN'
        previous=cues[-1] if cues else None
        split=(previous is None or previous['speaker']!=role
               or previous['source_turn_id']!=w.get('source_turn_id')
               or w['index']!=previous['last_word_index']+1
               or w['start']-previous['end']>1.5
               or w['end']-previous['start']>6.0 or len(previous['words'])>=20)
        if split:
            cues.append({'speaker':role,'start':w['start'],'end':w['end'],
                         'source_turn_id':w.get('source_turn_id'),'last_word_index':w['index'],'words':[]})
        cues[-1]['words'].append(str(w['text']).strip())
        cues[-1]['end']=w['end']
        cues[-1]['last_word_index']=w['index']
    return cues


def format_words(words, turn_style='plain'):
    if turn_style=='timestamped_cues':
        return '\n'.join(f"[{timestamp(c['start'])}] {c['speaker']}: "+' '.join(c['words'])
                         for c in cue_groups(words))
    turns=[]
    for w in words:
        role=w.get('speaker') if w.get('speaker') in ('P','T') else 'UNKNOWN'
        if not turns or turns[-1][0]!=role:
            turns.append((role,[]))
        turns[-1][1].append(str(w['text']).strip())
    return '\n'.join(f'{role}: '+' '.join(text) for role,text in turns)


def main(args):
    if (args.output/'cohere_new/replacement_audit.json').exists():
        raise ValueError('Dataset already prepared; use existing datasets or a fresh output root')
    release = args.release_root
    if not (release/'master_v1').is_dir():
        children = list(release.glob('*/master_v1'))
        if len(children)!=1: raise ValueError('Cannot uniquely locate release master_v1')
        release=children[0].parent
    summary=load(release/'summary.json')
    if not summary['complete']: raise ValueError('Incomplete source release')
    original=rows(args.original_master/'paired_master_soft.jsonl')
    if len(original)!=4325: raise ValueError('Expected original 4,325-segment cohort')
    old={r['segment_uid']:r for r in original}
    if len(old)!=len(original): raise ValueError('Duplicate original IDs')
    cache={}; updates={}; excluded=[]; source_hashes={}; session_repairs={}; session_review={}
    for row in original:
        sid=Path(row['video']).stem
        if sid not in cache:
            p=release/f'master_v1/{sid}/{sid}.words.json'
            if not p.exists(): cache[sid]=None
            else:
                data=load(p)
                if data.get('text_preserved') is not True: raise ValueError(f'{sid}: source text preservation not asserted')
                by_index={w['index']:w for w in data['words']}
                overlay=release/f'timed_dialogue_v1/{sid}/word_updates.json'
                overlay_words=load(overlay)['words']
                for word in overlay_words:
                    if word['index'] not in by_index or word['text'] != by_index[word['index']]['text']:
                        raise ValueError(f'{sid}: overlay changes wording/index')
                    by_index[word['index']]=word
                all_words=[by_index[i] for i in sorted(by_index)]
                if args.fill_unknown_roles:
                    if args.role_fill_policy=='contextual':
                        all_words,repairs,review=repair_contextual_roles(all_words)
                        session_review[sid]=review
                    else:
                        all_words,repairs=repair_unknown_roles(all_words)
                    session_repairs[sid]=repairs
                turns_path=release/f'timed_dialogue_v1/{sid}/{sid}.turns.json'
                for turn in load(turns_path)['turns']:
                    for idx in range(turn['first_word_index'],turn['last_word_index']+1):
                        if idx in by_index:
                            by_index[idx]['source_turn_id']=turn['turn_id']
                for word in all_words:
                    word['source_turn_id']=by_index[word['index']].get('source_turn_id')
                cache[sid]=[w for w in all_words if valid_word(w)]
                source_hashes[sid]={'words_sha256':sha(p),'overlay_sha256':sha(overlay)}
        if cache[sid] is None:
            excluded.append({'segment_uid':row['segment_uid'],'session_id':sid,'reason':'session_not_in_release'})
            continue
        selected=[w for w in cache[sid] if float(row['start_sec']) <= (w['start']+w['end'])/2 < float(row['end_sec'])]
        if not selected:
            excluded.append({'segment_uid':row['segment_uid'],'session_id':sid,'reason':'no_timed_words_in_rating_window'})
            continue
        unknown=sum(w.get('speaker') not in ('P','T') for w in selected)
        updates[row['segment_uid']]={
            'transcript_text':format_words(selected,args.turn_style),
            'transcript_text_plain':' '.join(str(w['text']).strip() for w in selected),
            'transcript_provider':'cohere_finetuned', 'transcript_status':'provisional_timed_roles',
            'llm_ready':True, 'transcript_timing_verified':False, 'transcript_roles_verified':False,
            'transcript_word_count':len(selected), 'transcript_unknown_role_words':unknown,
            'transcript_no_patient_role':not any(w.get('speaker')=='P' for w in selected),
            'transcript_inferred_role_words':sum(bool(w.get('dataset_role_inference')) for w in selected),
            'transcript_turn_style':args.turn_style,
            'review_flags':'Cohere timing and speaker roles provisional; UNKNOWN words retained',
            'transcript_source_hashes':source_hashes[sid]}
    kept=set(updates); test_counts=Counter(); fold_data=[]; fold_hashes={}
    for fold in range(1,6):
        path=args.original_master/f'fold_{fold}/master_manifest.jsonl'
        all_rows=rows(path)
        if {r['segment_uid'] for r in all_rows} != set(old): raise ValueError('Original fold cohort mismatch')
        selected=[r for r in all_rows if r['segment_uid'] in kept]
        patients={s:{r['patient_id'] for r in selected if r['split']==s} for s in ('train','val','test')}
        if any(patients[a]&patients[b] for a,b in [('train','val'),('train','test'),('val','test')]): raise ValueError('Patient leakage')
        if any(not values for values in patients.values()): raise ValueError('Empty split')
        test_counts.update(r['segment_uid'] for r in selected if r['split']=='test')
        fold_data.append((fold,selected)); fold_hashes[str(fold)]=sha(path)
    if set(test_counts)!=kept or set(test_counts.values())!={1}: raise ValueError('Test IDs must occur exactly once')
    audit={'rows':len(kept),'patients':len({old[u]['patient_id'] for u in kept}),
           'original_rows':4325,'subset_authorized':True,'source_release_sha256':sha(release/'release.json'),
           'original_fold_sha256':fold_hashes,'exclusions':dict(Counter(r['reason'] for r in excluded)),
           'minutes_with_unknown_roles':sum(v['transcript_unknown_role_words']>0 for v in updates.values()),
           'minutes_without_patient_role':sum(v['transcript_no_patient_role'] for v in updates.values()),
           'inferred_role_words_in_retained_minutes':sum(v['transcript_inferred_role_words'] for v in updates.values()),
           'minutes_with_inferred_roles':sum(v['transcript_inferred_role_words']>0 for v in updates.values()),
           'role_fill_method':('contextual_v2' if args.role_fill_policy=='contextual' else 'same_role_bracket_v1') if args.fill_unknown_roles else 'none',
           'turn_style':args.turn_style,
           'timing_verified':False,'roles_verified':False,
           'note':'Exploratory provisional source. Source release splits ignored; original patient folds retained.'}
    for condition in ('original_subset','cohere_new'):
        root=args.output/condition
        for fold, values in fold_data:
            output=[{**r,**updates[r['segment_uid']]} for r in values] if condition=='cohere_new' else values
            save_rows(root/f'fold_{fold}/master_manifest.jsonl',output)
        selected=[r for r in original if r['segment_uid'] in kept]
        save_rows(root/'paired_master_soft.jsonl', [{**r,**updates[r['segment_uid']]} for r in selected] if condition=='cohere_new' else selected)
        save(root/'replacement_audit.json',{**audit,'condition':condition})
    save(args.output/'preparation_summary.json',audit)
    save(args.output/'excluded_segments.json',excluded)
    save(args.output/'speaker_role_repairs.json',session_repairs)
    save(args.output/'speaker_role_review_candidates.json',session_review)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root',type=Path,required=True)
    p.add_argument('--original-master',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fill-unknown-roles',action='store_true')
    p.add_argument('--turn-style',choices=['plain','timestamped_cues'],default='plain')
    p.add_argument('--role-fill-policy',choices=['bounded','contextual'],default='bounded')
    main(p.parse_args())
