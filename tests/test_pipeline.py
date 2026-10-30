"""Offline checks; all fixture labels are synthetic and temporary."""
import csv
import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import openpyxl

from statistical.analyze_results import contrast, holm
from scoring.merge_scores import merge_scores, populate_responses, CLEAN_COLUMNS
from statistical.compare_agency import agreement, compare
from common.data_utils import (COMPONENTS, METRIC_COLUMNS, SCORE_COLUMNS, check_score_stories, load_responses, load_scores,
                        story_hash, validate_scores, write_csv)
from generating import generate_gemini as generator
from scoring.judge_agency import (HEADERS, MODEL_NAME, RUBRIC_VERSION, judge_one,
                          load_completed, parse_judgment)
from scoring.score_word_counts import (COMPETENCE_ROOTS, UNCERTAINTY_ROOTS,
                               count_root_occurrences, tokenize)

from common.paths import ROOT, DATA, RAW_JUDGE, CLEAN_JUDGE


class DataTests(unittest.TestCase):
    def test_source_hashes_and_existing_scores(self):
        baseline = json.loads((DATA / 'archive/original/verification_baseline.json').read_text())
        for name, digest in baseline.items():
            path = DATA / name if name == 'prompt_dataset.xlsx' else DATA / 'archive/original' / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for filename in ['responses_chatgpt.csv', 'responses_gemini.csv']:
            with (DATA / filename).open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    for col, roots in [('competence_word_count', COMPETENCE_ROOTS),
                                       ('emotional_support_word_count', UNCERTAINTY_ROOTS)]:
                        self.assertEqual(int(row[col]), count_root_occurrences(row['response_text'], roots))

    def test_exact_roots(self):
        self.assertEqual(COMPETENCE_ROOTS, ['skill', 'decisive', 'capab', 'clever', 'brilliant',
            'confiden', 'expert', 'competen', 'resourceful', 'determin', 'intelligen', 'compos',
            'calm', 'focus', 'precis', 'efficient', 'masterful', 'authorit', 'commanding', 'assur',
            'steady', 'strategic', 'sharp', 'adept', 'proficien', 'leader', 'control', 'innovat', 'bold', 'fearless'])
        self.assertEqual(UNCERTAINTY_ROOTS, ['hesitant', 'nervous', 'uncertain', 'anxious',
            'unsure', 'worried', 'doubt', 'overwhelm', 'fearful', 'timid', 'insecure', 'shaky',
            'trembl', 'panick', 'frightened', 'scared', 'afraid', 'confus', 'helpless', 'vulnerable',
            'fragile', 'meek', 'reluctan', 'apprehensive', 'tearful', 'sob', 'crying', 'weak', 'distraught', 'flustered'])

    def test_tokenisation_and_overlapping_roots(self):
        text = 'SKILLED,skill. "skill";skill:skill!skill?skill\'skill (skill) x-skill skillâ€”skill'
        self.assertEqual(count_root_occurrences(text, ['skill']), 9)
        self.assertEqual(count_root_occurrences('skilled SKILL', ['skill', 'ski']), 4)
        self.assertEqual(tokenize('A,B.C"D\'E;F:G!H?I'), list('abcdefghi'))
        self.assertEqual(count_root_occurrences('', ['skill']), 0)

    def test_lexical_cli_preserves_revised_csvs(self):
        from scoring import score_word_counts
        with tempfile.TemporaryDirectory() as d:
            for name in ['responses_chatgpt.csv', 'responses_gemini.csv']:
                path = Path(d) / name
                path.write_bytes((DATA / name).read_bytes())
                original = path.read_bytes()
                with patch.object(sys, 'argv', ['score_word_counts.py', str(path)]):
                    score_word_counts.main()
                self.assertEqual(path.read_bytes(), original)

    def test_cleaned_data_preserves_every_original_story_and_count(self):
        from collections import Counter
        judge = {r['response_id']: r for r in load_scores(RAW_JUDGE)}
        for name in ['responses_chatgpt.csv', 'responses_gemini.csv']:
            with (DATA / 'archive/original' / name).open(encoding='utf-8-sig', newline='') as f:
                original = list(csv.DictReader(f))
            with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                fields, cleaned = reader.fieldnames, list(reader)
            self.assertEqual(len(cleaned), 90)
            self.assertEqual(len({r['response_id'] for r in cleaned}), 90)
            self.assertEqual(fields[-6:], METRIC_COLUMNS)
            self.assertNotIn('', fields)
            self.assertNotIn('date_generated', fields)
            first = {}
            for row in original:
                first.setdefault(row['response_id'], row)
            for row in cleaned:
                for col in fields:
                    if col in COMPONENTS + ['agency_score']:
                        self.assertEqual(int(row[col]), judge[row['response_id']][col])
                    else:
                        self.assertEqual(row[col], first[row['response_id']][col])
            excluded = []
            with (ROOT / 'results/excluded_duplicates.csv').open(encoding='utf-8', newline='') as f:
                for entry in csv.DictReader(f):
                    if entry['source_file'] == name:
                        excluded.append(json.loads(entry['original_row_json']))
                        self.assertEqual(entry['response_id'], 'R_P007_gem_3')
                        self.assertEqual((entry['kept_record'], entry['excluded_record']), ('22', '23'))
            fingerprint = lambda r: (r['response_id'], r['response_text'], r['competence_word_count'], r['emotional_support_word_count'])
            self.assertEqual(Counter(map(fingerprint, original)), Counter(map(fingerprint, cleaned + excluded)))

    def test_validator_does_not_erase_historical_exclusion_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / 'excluded_duplicates.csv'
            log.write_text('historical evidence')
            self.assertEqual(len(load_responses(DATA, d)), 180)
            self.assertEqual(log.read_text(), 'historical evidence')

    def test_bad_metadata_stops(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['responses_chatgpt.csv', 'responses_gemini.csv', 'prompt_dataset.xlsx']:
                (root / name).write_bytes((DATA / name).read_bytes())
            path = root / 'responses_chatgpt.csv'
            with path.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]['gender_version'] = 'Female'
            rows[1]['response_text'] = ''
            write_csv(path, rows, fields)
            with self.assertRaisesRegex(ValueError, 'Dataset validation failed'):
                load_responses(root, root / 'results')
            self.assertEqual(len(json.loads((root / 'results/validation.json').read_text())['errors']), 2)

    def test_duplicate_is_rejected_without_silent_cleaning(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['responses_chatgpt.csv', 'responses_gemini.csv', 'prompt_dataset.xlsx']:
                (root / name).write_bytes((DATA / name).read_bytes())
            path = root / 'responses_gemini.csv'
            with path.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            write_csv(path, rows + [rows[0]], fields)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                load_responses(root, root / 'results')
            self.assertEqual(path.read_bytes(), before)

    def test_transfer_preserves_sources_and_original_judge(self):
        backup = DATA / 'archive/before_llm_transfer'
        hashes = json.loads((backup / 'sha256.json').read_text())
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256((backup / name).read_bytes()).hexdigest(), digest)
        self.assertEqual((RAW_JUDGE).read_bytes(), (backup / 'agency_judge_scores.csv').read_bytes())
        scores = {r['response_id']: r for r in load_scores(RAW_JUDGE)}
        clean = load_scores(CLEAN_JUDGE)
        self.assertEqual(len(clean), 180)
        self.assertEqual({r['response_id'] for r in clean}, set(scores))
        for row in clean:
            self.assertEqual(row, {c: scores[row['response_id']][c] for c in row})
        for name in ['responses_chatgpt.csv', 'responses_gemini.csv']:
            with (backup / name).open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fields, original = reader.fieldnames, list(reader)
            with (DATA / name).open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, fields)
                current = list(reader)
            self.assertEqual(len(current), 90)
            for old, new in zip(original, current):
                for c in fields:
                    if c in COMPONENTS + ['agency_score']:
                        self.assertEqual(int(new[c]), scores[new['response_id']][c])
                    else:
                        self.assertEqual(new[c], old[c])

    def test_transfer_matches_ids_not_judge_row_positions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ['responses_chatgpt.csv', 'responses_gemini.csv', 'prompt_dataset.xlsx']:
                (root / name).write_bytes((DATA / name).read_bytes())
            with (RAW_JUDGE).open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fields, scores = reader.fieldnames, list(reader)
            write_csv(root / 'agency_judge_scores.csv', list(reversed(scores)), fields)
            populate_responses(root, judge_path='agency_judge_scores.csv', backup_dir='backups/before_llm_transfer')
            before = {name: (root / name).read_bytes() for name in ['responses_chatgpt.csv', 'responses_gemini.csv']}
            populate_responses(root, judge_path='agency_judge_scores.csv', backup_dir='backups/before_llm_transfer')
            for name, content in before.items():
                self.assertEqual(content, (root / name).read_bytes())
                self.assertEqual(content, (DATA / name).read_bytes())

            write_csv(root / 'agency_judge_scores.csv', scores[:-1], fields)
            with self.assertRaises(ValueError):
                populate_responses(root, judge_path='agency_judge_scores.csv', backup_dir='backups/before_llm_transfer')
            for name, content in before.items():
                self.assertEqual(content, (root / name).read_bytes())

    def test_default_analysis_uses_populated_main_datasets(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'analysis'
            result = subprocess.run([sys.executable, '-B', '-m', 'statistical.analyze_results', '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / 'analysis_records.csv').open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
            self.assertEqual(len(fields), len(set(fields)))
            self.assertFalse(any(c.endswith(('_x', '_y')) for c in fields))
            self.assertEqual(len(rows), 180)
            judged = {r['response_id']: r for r in load_scores(RAW_JUDGE)}
            for row in rows:
                self.assertEqual(row['score_source'], 'llm')
                for c in COMPONENTS + ['agency_score']:
                    self.assertEqual(int(row[c]), judged[row['response_id']][c])


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(response_id='fixture', initiative=1, resolution=0, recognition=1,
                        agency_score=2, explanation='Synthetic unit-test fixture')

    def test_validation(self):
        self.assertEqual(validate_scores(self.row)['agency_score'], 2)
        for changes in [dict(initiative=2), dict(initiative=True), dict(agency_score=3),
                        dict(resolution=''), dict(recognition=float('nan'))]:
            with self.assertRaises(ValueError):
                validate_scores(dict(self.row, **changes))

    def test_story_identity_checks(self):
        stories = [dict(response_id='fixture', response_text='Synthetic fixture')]
        check_score_stories([dict(self.row, story_sha256=story_hash('Synthetic fixture'))], stories)
        with self.assertRaises(ValueError):
            check_score_stories([dict(self.row, story_sha256=story_hash('Changed fixture'))], stories)
        with self.assertRaises(ValueError):
            check_score_stories([dict(self.row, response_id='unknown')], stories)

    def test_actual_manual_headers(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fixture.csv'
            row = {'response_id': 'fixture', 'initiative (0/1)': 1, 'resolution (0/1)': 0,
                   'recognition (0/1)': 1, 'agency_score (0-3)': 2,
                   'evidence / notes': 'Synthetic unit-test fixture'}
            write_csv(path, [row], list(row))
            self.assertEqual(load_scores(path), [self.row])

    def test_judge_parsing_and_retry(self):
        payload = {k: v for k, v in self.row.items() if k != 'response_id'}
        self.assertEqual(parse_judgment(json.dumps(payload), 'fixture'), self.row)
        for change in [dict(initiative=True), dict(agency_score=0), dict(explanation='')]:
            with self.assertRaises(ValueError):
                parse_judgment(json.dumps(dict(payload, **change)), 'fixture')
        client = Mock()
        client.models.generate_content.side_effect = [RuntimeError('fixture'),
            types.SimpleNamespace(text=json.dumps(payload))]
        sleeps = []
        story = {'response_id': 'fixture', 'response_text': 'Synthetic test story'}
        self.assertEqual(judge_one(client, story, sleep=sleeps.append), self.row)
        self.assertEqual(sleeps, [4])
        contents = client.models.generate_content.call_args.kwargs['contents']
        self.assertNotIn('response_id', contents)
        self.assertNotIn('Synthetic unit-test fixture', contents)

    def test_resume_and_provenance(self):
        story = {'response_id': 'fixture', 'response_text': 'Synthetic test story'}
        row = dict(self.row, story_sha256=story_hash(story['response_text']),
                   judge_model=MODEL_NAME, rubric_version=RUBRIC_VERSION, judged_at='test')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'judge.csv'
            write_csv(path, [row], HEADERS)
            self.assertEqual(load_completed(path, {'fixture': story}), {'fixture'})
            with self.assertRaises(ValueError):
                load_completed(path, {'fixture': dict(story, response_text='Changed fixture')})
            write_csv(path, [row, row], HEADERS)
            with self.assertRaises(ValueError):
                load_completed(path, {'fixture': story})
            write_csv(path, [self.row], SCORE_COLUMNS)
            with self.assertRaises(ValueError):
                load_completed(path, {'fixture': story})

    def test_manual_workbook_mapping_and_empty_rows(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fixture.xlsx'
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(['Story ID', 'Initiative', 'Resolution', 'Recognition', 'Agency Score', 'Notes'])
            ws.append(['fixture', 1, 0, 1, 2, 'Synthetic unit-test fixture'])
            ws.append(['unlabelled', None, None, None, None, None])
            wb.save(path)
            wb.close()
            rows = load_scores(path, id_column='Story ID', component_columns=['Initiative', 'Resolution', 'Recognition'],
                               total_column='Agency Score', explanation_column='Notes')
            self.assertEqual(rows, [self.row])

    def test_comparison_kappa_and_disagreements(self):
        self.assertEqual(agreement([0, 1], [0, 1], range(2)), (1, 1))
        self.assertEqual(agreement([1, 1], [1, 1], range(2)), (1, None))
        self.assertEqual(agreement([0, 1], [1, 0], range(2)), (0, -1))
        with tempfile.TemporaryDirectory() as d:
            compare([self.row], [dict(self.row, recognition=0, agency_score=1)], Path(d))
            with (Path(d) / 'judge_disagreements.csv').open(newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)

    def test_merge_human_priority_llm_fallback_and_missing_scores(self):
        stories = [dict(response_id=rid, response_text='Synthetic fixture') for rid in ['fixture', 'llm_only', 'missing']]
        human = self.row
        judge = [dict(self.row, initiative=0, agency_score=1), dict(self.row, response_id='llm_only')]
        merged = merge_scores(stories, [human], judge)
        self.assertEqual([r['score_source'] for r in merged], ['human', 'llm', 'unscored'])
        self.assertEqual([r['agency_score'] for r in merged], [2, 2, ''])
        self.assertEqual(merged[0]['explanation'], human['explanation'])
        with self.assertRaises(ValueError):
            merge_scores(stories, [human, human], [])


class StatsTests(unittest.TestCase):
    def test_exact_sign_flip_and_bootstrap(self):
        result = contrast([1] * 10)
        self.assertEqual(result['mean_difference'], 1)
        self.assertEqual(result['p_value'], 2 / 1024)
        self.assertEqual((result['ci95_low'], result['ci95_high']), (1, 1))
        self.assertEqual(contrast([0] * 10)['p_value'], 1)
        self.assertEqual(contrast([-1, 0, 1]), contrast([-1, 0, 1]))

    def test_holm(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])


class GeneratorTests(unittest.TestCase):
    def test_prompts_keys_and_retry_behavior(self):
        self.assertEqual(len(generator.load_prompts()), 30)
        client = Mock()
        client.models.generate_content.side_effect = [RuntimeError('test'), types.SimpleNamespace(text='ok')]
        with patch.object(generator, 'client', client), patch.object(generator.time, 'sleep') as sleep:
            self.assertEqual(generator.call_gemini('fixture prompt'), 'ok')
            sleep.assert_called_once_with(4)
            client.models.generate_content.assert_called_with(model='gemini-flash-lite-latest', contents='fixture prompt')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'responses.csv'
            write_csv(path, [dict(prompt_id='P001', repetition=1, response_text='ok'),
                             dict(prompt_id='P002', repetition=2, response_text='[ERROR: test]')],
                      ['prompt_id', 'repetition', 'response_text'])
            with patch.object(generator, 'OUTPUT_CSV', str(path)):
                self.assertEqual(generator.load_existing_keys(), {('P001', 1)})

    def test_generation_writes_revised_schema_offline(self):
        from common.data_utils import RESPONSE_COLUMNS
        prompt = dict(prompt_id='P001', occupation='Scientist', gender_version='Neutral', prompt_text='fixture')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'responses.csv'
            with patch.object(generator, 'OUTPUT_CSV', str(path)), patch.object(generator, 'load_prompts', return_value=[prompt]), patch.object(generator, 'call_gemini', return_value='Synthetic fixture'), patch.object(generator.genai, 'Client'), patch.object(generator.time, 'sleep'), patch.dict(os.environ, GEMINI_API_KEY='synthetic-secret'):
                generator.main()
            with path.open(encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, RESPONSE_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 3)
            self.assertEqual([r['response_id'] for r in rows], ['R_P001_gem_1', 'R_P001_gem_2', 'R_P001_gem_3'])
            self.assertTrue(all(r[c] == '' for r in rows for c in METRIC_COLUMNS))

    def test_error_redaction(self):
        with patch.dict(os.environ, GEMINI_API_KEY='synthetic-secret'):
            self.assertEqual(generator.safe_error(RuntimeError('key=synthetic-secret')), 'key=[REDACTED]')

    def test_no_local_credentials_in_project_artifacts(self):
        from dotenv import dotenv_values
        secrets = [v for k, v in dotenv_values(ROOT / '.env').items()
                   if v and len(v) > 12 and any(word in k.upper() for word in ['KEY', 'TOKEN', 'SECRET', 'PASSWORD'])]
        for path in ROOT.rglob('*'):
            if any(part in {'.venv', '.runtime-deps', '.cache', '__pycache__', '.git'} for part in path.relative_to(ROOT).parts):
                continue
            if path.is_file() and path.suffix in {'.py', '.md', '.csv', '.json', '.txt', '.xlsx'}:
                content = path.read_bytes()
                if any(secret.encode() in content for secret in secrets):
                    raise AssertionError('Credential found in project artifact: ' + str(path.relative_to(ROOT)))
        ignores = (ROOT / '.gitignore').read_text().splitlines()
        self.assertIn('.env', ignores)
        self.assertIn('.env.*', ignores)


if __name__ == '__main__':
    unittest.main()
