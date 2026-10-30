"""Compare actual manual labels against judge scores by response_id (CSV/XLSX)."""
import argparse
from common.paths import CLEAN_JUDGE, RESULTS
from pathlib import Path

from common.data_utils import (COMPONENTS, add_score_arguments, check_score_stories,
                        load_responses, load_scores, score_options, write_csv)


def agreement(human, judge, categories):
    n = len(human)
    observed = sum(a == b for a, b in zip(human, judge)) / n
    chance = sum((human.count(c) / n) * (judge.count(c) / n) for c in categories)
    kappa = (observed - chance) / (1 - chance) if chance < 1 else None
    return observed, kappa


def compare(manual, automated, output):
    human = {r['response_id']: r for r in manual}
    judge = {r['response_id']: r for r in automated}
    shared = sorted(human.keys() & judge.keys())
    coverage = [dict(response_id=rid, status='matched' if rid in shared else
                     'manual_only' if rid in human else 'judge_only')
                for rid in sorted(human.keys() | judge.keys())]
    write_csv(output / 'validation_coverage.csv', coverage, ['response_id', 'status'])
    if not shared:
        raise ValueError('No manually labelled IDs overlap judge IDs. Judge is not validated.')
    metrics = []
    for column in ['agency_score', *COMPONENTS]:
        a = [human[r][column] for r in shared]
        b = [judge[r][column] for r in shared]
        exact, kappa = agreement(a, b, range(4) if column == 'agency_score' else range(2))
        metrics.append(dict(metric=column, n=len(shared), exact_agreement=exact,
                            mean_absolute_difference=sum(abs(x-y) for x, y in zip(a, b)) / len(a),
                            cohen_kappa=kappa,
                            kappa_note='undefined: expected agreement is 1' if kappa is None else 'unweighted'))
    write_csv(output / 'judge_agreement.csv', metrics,
              ['metric', 'n', 'exact_agreement', 'mean_absolute_difference', 'cohen_kappa', 'kappa_note'])
    disagreements = []
    columns = ['agency_score', *COMPONENTS]
    for rid in shared:
        if any(human[rid][c] != judge[rid][c] for c in columns):
            row = dict(response_id=rid)
            for source, scores in [('human', human), ('judge', judge)]:
                row.update({f'{source}_{c}': scores[rid].get(c, '') for c in columns + ['explanation']})
            disagreements.append(row)
    write_csv(output / 'judge_disagreements.csv', disagreements,
              ['response_id'] + [f'{s}_{c}' for s in ['human', 'judge'] for c in columns + ['explanation']])
    print(f'Compared {len(shared)} actual manually labelled stories. Outputs: {output}')
    print('Agreement is evidence for this labelled subset, not automatic validation of all stories.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual', required=True, help='Manual CSV or XLSX')
    parser.add_argument('--judge', default=str(CLEAN_JUDGE))
    parser.add_argument('--output', default=str(RESULTS / "judge_validation"))
    add_score_arguments(parser)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stories = load_responses(output=out.parent)
    manual = load_scores(args.manual, **score_options(args))
    judge = load_scores(args.judge)
    check_score_stories(manual + judge, stories)
    compare(manual, judge, out)


if __name__ == '__main__':
    main()
