# Narrative Bias in Generative AI

A comparison of agency, competence language, and uncertainty language in 180
ChatGPT and Gemini stories: 10 occupations x three gender conditions (Neutral,
Male, Female) x three generations x two models. Each model contributes 90 stories.
All stories have LLM agency scores; human annotations are kept separately for
validation. This experiment analyses text narratives.

## Setup and analysis

Use Python 3.12 or newer (verified with Python 3.14). From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python -m statistical.analyze_results
python -m statistical.compare_agency --manual data/human_reviewed_stories.xlsx
python -m scoring.judge_agency --dry-run
```

These checks and analyses run offline using the supplied data. Analysis tables
and four graphs are written to `results/analysis/`; human-versus-judge agreement,
coverage, and disagreements go to `results/judge_validation/`.

## Project layout

| Path | Contents |
| --- | --- |
| `data/responses_chatgpt.csv`, `data/responses_gemini.csv` | Stories, metadata, agency scores, and lexical counts |
| `data/prompt_dataset.xlsx` | Original prompts |
| `data/human_reviewed_stories.xlsx` | 40 completed human annotations, 20 per model |
| `data/agency_judge_scores_clean.csv` | Judge scores and explanations for all 180 stories |
| `data/archive/original/` | Initial responses and verification hashes |
| `data/archive/before_llm_transfer/` | Responses before score transfer, full judge output, and hashes |
| `generating/` | Gemini story generation |
| `scoring/` | Lexical scoring, agency judging, and score transfer |
| `statistical/` | Analysis and human agreement comparison |
| `common/` | Shared paths, loaders, and validation |
| `tests/` | Offline tests, including mocked API calls |

The full archived judge CSV retains narrative hashes, model/rubric identifiers,
and original timestamps for reproducibility and safe resume. The clean CSV is the
working copy. Local `.env` credentials, environments, editor settings, and caches
are ignored by Git.

## Scoring and validation

Agency is the sum of three binary criteria: **initiative** (starts the central
action), **resolution** (decisively helps resolve the problem), and **recognition**
(explicitly receives credit). Totals range from 0 to 3. The complete rubric lives
in `scoring/judge_agency.py`.

Lexical scores count tokens starting with the competence and uncertainty roots
in `scoring/score_word_counts.py`. Counting lowercases text, replaces only
`, . " ' ; : ! ?` with spaces, then splits on whitespace. Overlapping roots count
separately. The legacy `emotional_support_word_count` column stores uncertainty
counts. These are lexical indicators, not semantic classifications. Rates are
counts per 100 tokens, with each story weighted equally.

To recompute lexical counts or transfer the original judge scores by exact ID:

```powershell
python -m scoring.score_word_counts data/responses_chatgpt.csv
python -m scoring.score_word_counts data/responses_gemini.csv
python -m scoring.merge_scores
```

Transfer validates all 180 scores, preserves backups, and rejects conflicts with
existing scores. Human labels are not used in the main response files. For a
separate analysis giving complete human labels priority over LLM labels:

```powershell
python -m scoring.merge_scores --hybrid
python -m statistical.analyze_results --scores results/hybrid_scores.csv --output results/hybrid_analysis
```

Hybrid rows identify their source as `human`, `llm`, or `unscored`; components
are never mixed between annotators. Missing scores remain blank.

Loaders validate IDs, prompt mappings, model labels, repetitions, narratives,
schemas, and score totals. Analysis also checks existing lexical counts against
recomputation. Unexpected duplicates fail validation. Gemini originally contained
two narratives for `R_P007_gem_3`; the first CSV occurrence was retained without
selection by quality or score. The excluded row is in
`results/excluded_duplicates.csv`, and both originals remain archived.

Human agreement uses exact story IDs and reports total/component agreement, mean
absolute difference, unweighted Cohen's kappa, coverage, and disagreements. Kappa
is undefined when expected agreement is one. Annotated text or hashes, when
provided, must match the source. Partial labels, invalid totals, duplicate or
unknown IDs fail validation. CSV and XLSX annotations support column overrides
via `--sheet`, `--id-column`, `--component-columns`, `--total-column`, and
`--explanation-column`; Excel formulas need saved cached values.

## Statistical method

Outputs include counts, per-story values, means, sample SDs, agency distributions,
component rates, occupation differences, and statistical tests. The four plots
show agency means, agency component rates, competence frequency, and uncertainty
frequency by gender condition and model.

The three generations are averaged within each model x occupation x gender cell.
Male - Female is the primary agency contrast; Neutral - Male and Neutral - Female
are secondary. Comparisons use within-occupation differences, equally weighting
the ten occupations. Repetition numbers are not matched story pairs. Model
comparisons first average gender cells within each occupation. Occupation-specific
differences are descriptive, without tests based on only three stories per cell.

Two-sided tests enumerate all 2^10 sign flips of occupation contrasts, assuming
exchangeability under the null or symmetric independent contrasts. Approximate
95% percentile intervals resample whole occupations 20,000 times with a fixed
seed recorded alongside input hashes and package versions in `run_info.json`.
The effect size is the raw mean difference. Holm correction uses three families:
the two primary agency tests, all remaining gender tests, and model comparisons.
Raw lexical counts and normalised rates share the secondary family; story length
is descriptive. Incomplete scores are summarised but excluded from inference.

Ten chosen occupations limit precision and generalisation; intervals can be
unstable or degenerate and are not simultaneous. Results are conditional on these
prompts and scoring assumptions, not causal effects on people or claims about all
multimodal outputs. Agreement on 40 selected stories does not validate every
judge score. Gemini judging its own stories may introduce evaluator bias; inspect
the agreement reports before interpreting agency findings. Hybrid scores use a
different measurement mix and should be reported separately.

## Optional API workflow

Existing data is sufficient for the offline workflow. To generate or judge stories,
set `GEMINI_API_KEY` in the environment or a root `.env` file:

```powershell
python -m generating.generate_gemini
python -m scoring.judge_agency --output results/new_judge_scores.csv
```

Both scripts use `gemini-flash-lite-latest`. Generation resumes completed
prompt/repetition keys. The judge sends the rubric and story text, validates JSON,
saves incrementally, and resumes only when IDs, story hashes, model, and rubric
match. It stops after persistent failures; `--delay` and `--limit` control new
judgements. The default resume file is the complete archived original; use a new
output path for fresh scores. Model aliases can change upstream, so future API
outputs may differ even with pinned dependencies.
