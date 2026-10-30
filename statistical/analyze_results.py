"""Descriptive analysis and occupation-level comparisons; never calls an API."""
import argparse
from common.paths import ROOT, DATA, RESULTS
import hashlib
import itertools
import json
import os
from pathlib import Path


os.environ.setdefault('MPLCONFIGDIR', str((ROOT / '.venv/cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data_utils import (COMPONENTS, add_score_arguments, check_score_stories,
                        load_responses, load_scores, score_options, validate_scores)
from scoring.score_word_counts import (COMPETENCE_ROOTS, UNCERTAINTY_ROOTS,
                               count_root_occurrences, tokenize)

GENDERS = ['Neutral', 'Male', 'Female']
LEXICAL = ['competence_word_count', 'emotional_support_word_count']
RATES = [c + '_per_100_words' for c in LEXICAL]


def holm(p_values):
    """Holm step-down family-wise correction, retaining original row order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def contrast(differences, seed=20261030):
    """Raw mean difference effect size; exact two-sided occupation sign flips.

    The 3 generations are averaged first. Repetition 1 in one condition is NOT
    paired to repetition 1 in another. Whole occupations are bootstrap units.
    Percentile CIs are approximate, especially with only ten occupations.
    """
    d = np.asarray(differences, dtype=float)
    if len(d) < 2 or not np.isfinite(d).all():
        raise ValueError('At least two finite occupation contrasts are needed')
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    estimate = d.mean()
    p = np.mean(np.abs((signs * d).mean(axis=1)) >= abs(estimate) - 1e-12)
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(20000, len(d)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [.025, .975])
    return dict(n_occupations=len(d), mean_difference=estimate, ci95_low=low,
                ci95_high=high, p_value=p,
                method='exact occupation sign-flip; percentile occupation bootstrap CI')


def summaries(df, metrics, out):
    for groups, filename in [(['model', 'gender_version'], 'summary_statistics.csv'),
                             (['model', 'occupation', 'gender_version'], 'occupation_summary.csv')]:
        tables = []
        for metric in metrics:
            table = df.groupby(groups)[metric].agg(n='count', mean='mean', standard_deviation='std').reset_index()
            table['metric'] = metric
            tables.append(table)
        pd.concat(tables, ignore_index=True).to_csv(out / filename, index=False)
    if 'agency_score' in metrics:
        labelled = df.dropna(subset=['agency_score'])
        distribution = pd.crosstab([labelled.model, labelled.gender_version], labelled.agency_score)
        distribution.reindex(columns=[0, 1, 2, 3], fill_value=0).to_csv(out / 'agency_score_distribution.csv')


def comparisons(df, metrics, out):
    results, occupations, skipped = [], [], []
    for metric in metrics:


        if df[metric].isna().any():
            skipped.append(f'{metric}: inference skipped; scores are incomplete.')
            continue
        means = df.groupby(['model', 'occupation', 'gender_version'])[metric].mean()
        for model in sorted(df.model.unique()):
            table = means.loc[model].unstack('gender_version')
            for left, right in [('Male', 'Female'), ('Neutral', 'Male'), ('Neutral', 'Female')]:
                d = table[left] - table[right]
                family = ('primary_agency' if metric == 'agency_score' and left == 'Male'
                          else 'secondary_gender')
                results.append(dict(metric=metric, model=model, comparison=f'{left} - {right}',
                                    family=family, **contrast(d)))
                for occupation, value in d.items():
                    occupations.append(dict(metric=metric, model=model, occupation=occupation,
                                            comparison=f'{left} - {right}', mean_difference=value,
                                            note='Exploratory description; 3 stories per condition, no paired repetitions'))
        # Equal weight to gender conditions, then match the two models by occupation.
        model_means = means.groupby(['model', 'occupation']).mean().unstack('model')
        results.append(dict(metric=metric, model='Both', comparison='ChatGPT - Gemini',
                            family='model_comparison',
                            **contrast(model_means.ChatGPT - model_means.Gemini)))
    table = pd.DataFrame(results)
    if not table.empty:
        table['p_holm'] = table.groupby('family')['p_value'].transform(lambda p: holm(p))
    table.to_csv(out / 'statistical_tests.csv', index=False)
    pd.DataFrame(occupations).to_csv(out / 'occupation_differences.csv', index=False)
    return skipped


def plot_metric(df, metric, ylabel, filename, out):
    table = df.groupby(['gender_version', 'model'])[metric].mean().unstack('model').reindex(GENDERS)
    ax = table.plot.bar(color=['#2878a0', '#dd8c35'], rot=0, figsize=(7.5, 4.5))
    ax.set(xlabel='Prompt gender condition', ylabel=ylabel)
    ax.set_title('Observed story means' + (' (available labels)' if metric == 'agency_score' else ''))
    if metric == 'agency_score':
        ax.set_ylim(0, 3)
    ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1.01, 1))
    # Pandas draws NaNs as absent bars. Label them so they cannot be read as zero.
    for container, model in zip(ax.containers, table.columns):
        for bar, value in zip(container, table[model]):
            if pd.isna(value):
                ax.text(bar.get_x() + bar.get_width()/2, .02, 'No labels', rotation=90,
                        ha='center', va='bottom', fontsize=8)
    ax.grid(axis='y', alpha=.2)
    plt.tight_layout()
    plt.savefig(out / filename, dpi=180)
    plt.close()


def plots(df, out):
    for metric, title, filename in zip(RATES,
            ['Competence roots per 100 words', 'Uncertainty roots per 100 words'],
            ['competence_frequency.png', 'uncertainty_frequency.png']):
        plot_metric(df, metric, title, filename, out)
    if 'agency_score' not in df or not df.agency_score.notna().any():
        return
    plot_metric(df, 'agency_score', 'Mean Agency Score (0-3)', 'agency_by_gender_model.png', out)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, ['ChatGPT', 'Gemini']):
        table = df[df.model == model].groupby('gender_version')[COMPONENTS].mean().reindex(GENDERS)
        table.plot.bar(
            ax=ax, rot=0, color=['#2878a0', '#dd8c35', '#4f9569'])
        ax.set(title=model, xlabel='Prompt gender condition', ylim=(0, 1))
        ax.set_ylabel('Proportion scored 1 (available labels)')
        ax.get_legend().remove()
        for container, component in zip(ax.containers, table.columns):
            for bar, value in zip(container, table[component]):
                if pd.isna(value):
                    ax.text(bar.get_x() + bar.get_width()/2, .02, 'No labels', rotation=90,
                            ha='center', va='bottom', fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(out / 'agency_component_rates.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', help='Optional explicit alternative score file; default uses main response CSVs')
    parser.add_argument('--output', default=str(RESULTS / "analysis"))
    add_score_arguments(parser)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for name in ['agency_by_gender_model.png', 'agency_component_rates.png', 'agency_score_distribution.csv']:
        (out / name).unlink(missing_ok=True)
    rows = load_responses(output=out.parent)
    df = pd.DataFrame(rows)

    df = df.drop(columns=[*COMPONENTS, 'agency_score'])
    df['repetition'] = df['repetition'].astype(int)
    df.groupby(['model', 'occupation', 'gender_version', 'repetition']).size().rename('n').to_csv(out / 'record_counts.csv')
    for column in ['model', 'occupation', 'gender_version', 'repetition']:
        df.groupby(column).size().rename('n').to_csv(out / f'counts_by_{column}.csv')
    df['word_count'] = df.response_text.map(lambda text: len(tokenize(text)))
    for column, roots, rate in zip(LEXICAL, [COMPETENCE_ROOTS, UNCERTAINTY_ROOTS], RATES):

        computed = df.response_text.map(lambda text: count_root_occurrences(text, roots))
        existing = pd.to_numeric(df[column], errors='coerce')
        mismatch = existing.notna() & (existing != computed)
        if mismatch.any():
            df.loc[mismatch, ['response_id', column]].to_csv(out / f'{column}_mismatches.csv', index=False)
            raise ValueError(f'Existing {column} disagrees with scorer; see mismatch log')
        df[column] = computed
        df[rate] = computed * 100 / df.word_count
    metrics = ['word_count', *LEXICAL, *RATES]
    status = []
    path = Path(args.scores) if args.scores else None
    if path is None:
        scores = []
        for row in rows:
            if any(row[c] != '' for c in COMPONENTS + ['agency_score']):
                score = validate_scores({c: row[c] for c in ['response_id', *COMPONENTS, 'agency_score']})
                score['score_source'] = 'llm'
                scores.append(score)
        status.append(f'{len(scores)}/180 LLM-labelled stories from the main response datasets. Human labels remain separate.')
    elif path.exists():
        scores = load_scores(path, **score_options(args))
        check_score_stories(scores, rows)
        status.append(f'{len(scores)}/180 labelled stories from explicit alternative: {path}.')
    else:
        scores = []
        status.append(f'Agency scores unavailable: {path}. Required columns: response_id, initiative, resolution, recognition, agency_score. Lexical analysis completed independently.')
    if scores:
        df = df.merge(pd.DataFrame(scores), on='response_id', how='left', validate='one_to_one')
        metrics += ['agency_score', *COMPONENTS]
        if 'score_source' in df:
            df['score_source'] = df['score_source'].fillna('unscored')
            df.groupby('score_source').size().rename('n').to_csv(out / 'score_source_counts.csv')
    missing = df[['response_id', 'model', 'occupation', 'gender_version']].copy()
    missing = missing if 'agency_score' not in df else missing[df.agency_score.isna()]
    missing.to_csv(out / 'unlabelled_stories.csv', index=False)
    summaries(df, metrics, out)

    status += comparisons(df, [m for m in metrics if m != 'word_count'], out)
    plots(df, out)
    df.drop(columns=['response_text', 'prompt_text_to_paste'], errors='ignore').to_csv(out / 'analysis_records.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps(dict(seed=20261030, bootstrap_samples=20000,
        scores_file=str(path) if path else 'main response datasets (all LLM)',
        scores_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path and path.exists() else None,
        response_sha256={name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in ['responses_chatgpt.csv', 'responses_gemini.csv']},
        status=status, versions=dict(numpy=np.__version__, pandas=pd.__version__,
        matplotlib=matplotlib.__version__)), indent=2), encoding='utf-8')
    (out / 'status.txt').write_text('\n'.join(status) + '\n', encoding='utf-8')
    print('\n'.join(status))
    print(f'Outputs: {out}')


if __name__ == '__main__':
    main()
