"""RQ1 descriptive gender distributions and exploratory exact tests (offline)."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from math import comb
import os
from pathlib import Path

from common.data_utils import write_csv
from common.paths import ROOT
from scoring.classify_gender import (LABELS, MODEL_NAME, OUTPUT, load_completed,
                                     select_neutral, write_review)

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.venv/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def binomial_exact(male, female):
    if any(type(x) is not int or x < 0 for x in [male, female]):
        raise ValueError('Counts must be nonnegative integers')
    n = male + female
    if not n:
        return None
    observed = comb(n, male)
    return Fraction(sum(comb(n, k) for k in range(n + 1) if comb(n, k) <= observed), 2 ** n)


def fisher_exact(table):
    if len(table) != 2 or any(len(row) != 2 for row in table):
        raise ValueError('Expected a 2x2 table')
    if any(type(x) is not int or x < 0 for row in table for x in row):
        raise ValueError('Counts must be nonnegative integers')
    (a, b), (c, d) = table
    r1, r2, male = a + b, c + d, a + c
    total = r1 + r2
    if not r1 or not r2 or not male or male == total:
        return None
    weights = {x: comb(r1, x) * comb(r2, male - x)
               for x in range(max(0, male - r2), min(r1, male) + 1)}
    return Fraction(sum(w for w in weights.values() if w <= weights[a]), comb(total, male))


def distributions(rows, groups):
    grouped = {}
    for row in rows:
        if row['gender_label'] not in LABELS:
            raise ValueError('Incomplete or invalid labels cannot enter analysis')
        key = tuple(row[g] for g in groups)
        grouped.setdefault(key, Counter())[row['gender_label']] += 1
    result = []
    for key, counts in sorted(grouped.items()):
        n = sum(counts.values())
        for label in LABELS:
            result.append(dict(zip(groups, key), gender_label=label, count=counts[label],
                               denominator=n, percentage=100 * counts[label] / n))
    return result


def exact_tests(rows):
    records, table = [], []
    for model in ['ChatGPT', 'Gemini']:
        selected = [r for r in rows if r['model'] == model]
        counts = Counter(r['gender_label'] for r in selected)
        if set(counts) - set(LABELS):
            raise ValueError('Invalid gender label')
        male, female = counts['male'], counts['female']
        table.append([male, female])
        p = binomial_exact(male, female)
        records.append(dict(test='two-sided exact binomial (p0=0.5)', model=model,
                            male=male, female=female, n_test=male + female,
                            excluded=len(selected) - male - female,
                            p_value=float(p) if p is not None else '',
                            p_exact_fraction=str(p) if p is not None else '',
                            status='computed' if p is not None else 'not estimable: no binary labels'))
    p = fisher_exact(table)
    records.append(dict(test='two-sided Fisher exact', model='ChatGPT vs Gemini',
                        male=sum(r[0] for r in table), female=sum(r[1] for r in table),
                        n_test=sum(map(sum, table)), excluded=len(rows) - sum(map(sum, table)),
                        p_value=float(p) if p is not None else '',
                        p_exact_fraction=str(p) if p is not None else '',
                        status='computed' if p is not None else 'not estimable: zero row/column margin'))
    return records, table


def charts(rows, output):
    colors = ['#2878a0', '#d6773a', '#7950a1', '#8c9299']
    names = ['Male', 'Female', 'Explicit other', 'Unknown / ambiguous']
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0., 0.]
    for label, color, name in zip(LABELS, colors, names):
        values = []
        for model in ['ChatGPT', 'Gemini']:
            selected = [r for r in rows if r['model'] == model]
            values.append(100 * sum(r['gender_label'] == label for r in selected) / len(selected))
        bars = ax.bar(['ChatGPT (n=30)', 'Gemini (n=30)'], values, bottom=bottom, color=color, label=name)
        for i, bar in enumerate(bars):
            if values[i]:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[i] + values[i] / 2,
                        f'{values[i]:.1f}%\n(n={round(values[i] * .3)})', ha='center', va='center', color='white')
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set(ylabel='Percentage of all neutral stories', ylim=(0, 100),
           title='Protagonist gender in neutral occupational stories')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(output / 'gender_by_model.png', dpi=180)
    plt.close(fig)
    occupations = sorted({r['occupation'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, model in zip(axes, ['ChatGPT', 'Gemini']):
        left = [0] * len(occupations)
        for label, color, name in zip(LABELS, colors, names):
            values = [sum(r['model'] == model and r['occupation'] == o and r['gender_label'] == label for r in rows)
                      for o in occupations]
            bars = ax.barh(occupations, values, left=left, color=color, label=name)
            for i, bar in enumerate(bars):
                if values[i]:
                    ax.text(left[i] + values[i] / 2, bar.get_y() + bar.get_height()/2,
                            str(values[i]), ha='center', va='center', color='white')
            left = [a + b for a, b in zip(left, values)]
        ax.set(title=model, xlabel='Stories per occupation (n=3)', xlim=(0, 3), xticks=[0, 1, 2, 3])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.suptitle('Neutral-story gender by occupation (descriptive only)')
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output / 'gender_by_occupation.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='RQ1 classification and analysis directory')
    parser.add_argument('--model', default=MODEL_NAME, help='Must match cached classifier model')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT.resolve()):
        parser.error('Output must be results/rq1 or a subdirectory of it')
    stories = select_neutral(output=output)
    completed = load_completed(output / 'classifications.jsonl', {r['response_id']: r for r in stories}, args.model)
    rows = write_review(stories, completed, output)
    if len(completed) != 60:
        raise SystemExit(f'Only {len(completed)}/60 classified. Run python -m scoring.classify_gender before analysis.')
    for groups, filename in [([], 'counts_overall.csv'), (['model'], 'counts_by_model.csv'),
                             (['occupation'], 'counts_by_occupation.csv'),
                             (['model', 'occupation'], 'counts_by_model_occupation.csv')]:
        summary = distributions(rows, groups)
        write_csv(output / filename, summary, list(summary[0]))
    tests, table = exact_tests(rows)
    write_csv(output / 'statistical_tests.csv', tests, list(tests[0]))
    charts(rows, output)
    report = dict(n_stories=len(rows), counts=distributions(rows, []), tests=tests,
                  fisher_table=dict(rows=['ChatGPT', 'Gemini'], columns=['male', 'female'], values=table),
                  review_needed=[r['response_id'] for r in rows if r['review_needed']],
                  interpretation='Exploratory unadjusted tests conditional on binary labels. These do not establish presence or absence of bias. Stories share occupational prompts; independence is an approximation. No occupation-level significance tests. LLM labels are not human ground truth; Gemini also classifies its own generated stories. Confidence is uncalibrated self-report, not a probability of correctness.')
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
