# RQ1: gender in neutral occupational stories

This workflow selects the existing 60 neutral stories (30 per generating model,
10 occupations, three runs each). It does not regenerate or alter source stories.

From the project root, using the installed environment:

```powershell
.venv\Scripts\python.exe -m scoring.classify_gender
.venv\Scripts\python.exe -m statistical.analyze_gender
.venv\Scripts\python.exe -m unittest discover -s tests
```

The classifier reads `GEMINI_API_KEY` from the environment or ignored root `.env`.
`--offline` exports the selected stories and any cached labels without API calls.
`--limit N` limits new classifications. Successful labels are saved incrementally;
rerunning resumes only when the story hashes, model, rubric, and schema match.
API/validation failures stop after three attempts without inventing a label.
Rate-limit/server retries wait at least 30 seconds. Error text and keys are not logged.

## Outputs

- `neutral_stories.csv`: selected IDs, generating models, occupations, runs, and text.
- `classifications.jsonl`: validated Gemini responses with source hashes and provenance.
- `classification_config.json`: complete rubric/schema and input hashes.
- `human_review.csv`: all labels, verbatim evidence, confidence, explanation, review
  flags, and original text. `review_needed.csv` contains just flagged rows.
- `counts_*.csv`: all four categories, including zero counts, with denominators and
  percentages overall, by model, by occupation, and by model within occupation.
- `statistical_tests.csv`: test sample sizes, excluded counts, p-values as decimals
  and exact rational fractions. `summary.json` also includes the Fisher table.
- `gender_by_model.png`, `gender_by_occupation.png`: descriptive comparisons.
- `failures.jsonl`, if present: historical failed attempts, not missing final labels.
- `initial_attempt/`: superseded v1 classifications/configuration/failure log, retained
  for audit only. These are never loaded into the final analysis. The first pass
  confused a student with the teacher and overlooked later pronouns in a nurse story;
  v2 explicitly requires identifying the occupational actor and reading the full text.

## Interpretation and review

Labels are `male`, `female`, `nonbinary_or_other_explicit`, or
`unknown_or_ambiguous`. Only protagonist pronouns or explicit gender references
support a gender label. Names, occupations, traits, and stereotypes are insufficient;
singular they alone does not establish nonbinary identity. Quotes must be exact
substrings, including markdown and punctuation. This mechanical check cannot prove
that a pronoun refers to the protagonist; readers must check its referent.

Review flags cover ambiguous/explicit-other labels, confidence below 0.90, mixed
male/female cues in quoted passages, and potentially inappropriate name/role-based
reasoning. Flags are screening heuristics, not validated measures of label error.
Confidence is uncalibrated model self-report. No human validation is claimed.

Two-sided binomial tests compare male/female labels within each model to 50:50.
Fisher's exact test compares the two models' male/female table, unless a row or
column margin is zero. Other/ambiguous labels remain in descriptive denominators
but are excluded from these tests. No occupation-level significance tests are run.
Exact integer enumeration uses probability ordering, consistent with the
[binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
and [Fisher test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)
definitions; no new dependency is required.

P-values are exploratory and unadjusted. Stories share occupational prompts, so
independence is an approximation; the occupations are not a random population sample.
Gemini also classifies its own stories. These tests do not establish the presence
or absence of bias. Results describe this dataset and the stated classification rule.
