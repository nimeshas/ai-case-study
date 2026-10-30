"""Offline LLM score transfer; optional separate hybrid scores with human priority."""
import argparse
import hashlib
import json
import shutil
from common.paths import DATA, RESULTS, RAW_JUDGE, CLEAN_JUDGE, MANUAL
from collections import Counter
from pathlib import Path

from common.data_utils import (COMPONENTS, SCORE_COLUMNS, SOURCES, add_score_arguments, check_score_stories,
                        load_responses, load_scores, score_options, story_hash, write_csv)
from scoring.score_word_counts import read_csv_robust

MERGED_COLUMNS = [*SCORE_COLUMNS, 'score_source', 'story_sha256']
CLEAN_COLUMNS = ['response_id', *COMPONENTS, 'agency_score', 'judge_reason']


def populate_responses(root=DATA, judge_path='archive/before_llm_transfer/agency_judge_scores.csv',
                       clean_path='agency_judge_scores_clean.csv',
                       backup_dir='archive/before_llm_transfer'):
    """Validate all inputs first, back up originals, then transfer by exact ID.

    Existing nonblank response scores must already agree: this is a transfer,
    never a rescore or correction. Human annotations are not read by this mode.
    """
    root = Path(root)
    judge_path, clean_path, backup_dir = [root / p for p in [judge_path, clean_path, backup_dir]]
    protected = {root / name for name, _, _ in SOURCES} | {judge_path, root / 'human_reviewed_stories.xlsx', root / 'prompt_dataset.xlsx'}
    if clean_path.resolve() in {p.resolve() for p in protected}:
        raise ValueError('Clean output must be separate from source and annotation files')
    if clean_path.resolve().is_relative_to(backup_dir.resolve()):
        raise ValueError('Clean output cannot overwrite preserved backups')
    stories = load_responses(root, root / 'results')
    scores = load_scores(judge_path)
    if any(row.get('score_source') not in (None, '', 'llm') for row in scores):
        raise ValueError('Transfer requires all-LLM scores; hybrid annotations must stay separate')
    check_score_stories(scores, stories)
    if len(scores) != 180 or {r['response_id'] for r in scores} != {r['response_id'] for r in stories}:
        raise ValueError('Judge scores must cover all 180 expected IDs exactly once')
    by_id = {r['response_id']: r for r in scores}
    staged = []
    for name, _, _ in SOURCES:
        fields, rows = read_csv_robust(root / name)
        for row in rows:
            score = by_id[row['response_id']]
            for column in COMPONENTS + ['agency_score']:
                if row[column] != '' and float(row[column]) != score[column]:
                    raise ValueError(f"Existing score differs for {row['response_id']}: {column}; nothing transferred")
                row[column] = str(score[column])
        staged.append((name, fields, rows))
    clean = [{**{c: row[c] for c in CLEAN_COLUMNS[:-1]}, 'judge_reason': row['explanation']}
             for row in scores]
    # Immutable, byte-exact originals; reruns keep the first backup.
    backup_dir.mkdir(parents=True, exist_ok=True)
    inputs = [root / name for name, _, _ in SOURCES] + [judge_path]
    for source in inputs:
        backup = backup_dir / source.name
        if backup.exists() and source == judge_path and backup.read_bytes() != source.read_bytes():
            raise ValueError('Judge output differs from preserved original; choose a new --backup-dir')
        if not backup.exists():
            shutil.copy2(source, backup)
    manifest = backup_dir / 'sha256.json'
    hashes = {p.name: hashlib.sha256((backup_dir / p.name).read_bytes()).hexdigest() for p in inputs}
    if manifest.exists() and json.loads(manifest.read_text()) != hashes:
        raise ValueError('Backup hash verification failed')
    manifest.write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    write_csv(clean_path, clean, CLEAN_COLUMNS)
    for name, fields, rows in staged:
        write_csv(root / name, rows, fields)
    print(f'Transferred 180 LLM scores by response_id. Clean judge file: {clean_path}. Backups: {backup_dir}')


def merge_scores(stories, manual, judge):
    """Return one row per story; missing labels stay blank, never zero-filled."""
    for scores in [manual, judge]:
        if len({r['response_id'] for r in scores}) != len(scores):
            raise ValueError('Duplicate annotation IDs')
        check_score_stories(scores, stories)
    human = {r['response_id']: r for r in manual}
    automated = {r['response_id']: r for r in judge}
    merged = []
    for story in stories:
        rid = story['response_id']
        row = {c: '' for c in MERGED_COLUMNS}
        row.update(response_id=rid, score_source='unscored', story_sha256=story_hash(story['response_text']))
        selected = human.get(rid) or automated.get(rid)
        if selected is not None:
            row.update({c: selected.get(c, '') for c in SCORE_COLUMNS})
            row['score_source'] = 'human' if rid in human else 'llm'
        merged.append(row)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hybrid', action='store_true', help='Write separate human-priority scores; never modify responses')
    parser.add_argument('--manual', default=str(MANUAL))
    parser.add_argument('--judge', default=str(RAW_JUDGE))
    parser.add_argument('--output', default=str(RESULTS / "hybrid_scores.csv"), help='Hybrid output only')
    parser.add_argument('--clean-output', default=str(CLEAN_JUDGE))
    parser.add_argument('--backup-dir', default=str(DATA / "archive/before_llm_transfer"))
    add_score_arguments(parser)
    args = parser.parse_args()
    if not args.hybrid:
        populate_responses(judge_path=args.judge, clean_path=args.clean_output, backup_dir=args.backup_dir)
        return
    output = Path(args.output)
    protected = [args.manual, args.judge, args.clean_output, str(CLEAN_JUDGE), str(DATA / 'responses_chatgpt.csv'), str(DATA / 'responses_gemini.csv'),
                 str(DATA / 'prompt_dataset.xlsx')]
    if output.suffix.lower() != '.csv' or output.resolve() in {Path(p).resolve() for p in protected}:
        parser.error('Output must be a separate merged CSV, not a source file')
    if output.exists():
        from scoring.score_word_counts import read_csv_robust
        columns, _ = read_csv_robust(output)
        if columns != MERGED_COLUMNS:
            parser.error('Existing output is not a merged scores file; choose another path')
    stories = load_responses()
    manual = load_scores(args.manual, **score_options(args))
    judge = load_scores(args.judge) if Path(args.judge).exists() else []
    if not Path(args.judge).exists():
        print(f'Judge scores unavailable: {args.judge}; remaining stories will stay unscored.')
    merged = merge_scores(stories, manual, judge)
    write_csv(output, merged, MERGED_COLUMNS)
    report = dict(manual_file=args.manual, judge_file=args.judge,
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in map(Path, [args.manual, args.judge, str(DATA / 'responses_chatgpt.csv'), str(DATA / 'responses_gemini.csv')])
                                if p.exists()},
                  counts=dict(Counter(r['score_source'] for r in merged)),
                  total_stories=len(merged), rule='Complete human scores override LLM scores; otherwise unscored')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Merged {len(merged)} stories: {report["counts"]}. Output: {output}')


if __name__ == '__main__':
    main()
