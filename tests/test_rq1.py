"""Offline RQ1 checks; API fixtures are synthetic and never saved as findings."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from common.paths import DATA
from common.data_utils import story_hash
from scoring.classify_gender import (
    LABELS, MODEL_NAME, RUBRIC_HASH, RUBRIC_VERSION, classify_one, load_completed,
    review_reasons, select_neutral, validate_label, write_review,
)
from statistical.analyze_gender import binomial_exact, distributions, exact_tests, fisher_exact


class SelectionTests(unittest.TestCase):
    def test_neutral_selection_preserves_ids_text_and_sources(self):
        paths = [DATA / n for n in ['responses_chatgpt.csv', 'responses_gemini.csv', 'prompt_dataset.xlsx']]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with tempfile.TemporaryDirectory() as d:
            rows = select_neutral(output=Path(d))
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({r['response_id'] for r in rows}), 60)
        for model in ['ChatGPT', 'Gemini']:
            selected = [r for r in rows if r['model'] == model]
            self.assertEqual(len(selected), 30)
            self.assertEqual(len({r['occupation'] for r in selected}), 10)
            for occupation in {r['occupation'] for r in selected}:
                self.assertEqual(sorted(r['repetition'] for r in selected if r['occupation'] == occupation), ['1', '2', '3'])
        self.assertTrue(all(r['gender_version'] == 'Neutral' for r in rows))
        self.assertIn('R_P001_gpt_1', {r['response_id'] for r in rows})
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def test_incomplete_or_duplicate_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows = select_neutral(output=Path(d))
        for invalid in [rows[:-1], rows[:-1] + [rows[0]]]:
            with patch('scoring.classify_gender.load_responses', return_value=invalid):
                with self.assertRaises(ValueError):
                    select_neutral()


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.quote = 'Alex inspected the pump. She repaired the broken valve.'
        self.story = dict(response_id='fixture', response_text=self.quote,
                          model='ChatGPT', occupation='Engineer', repetition='1')
        self.label = dict(response_id='fixture', category='female', evidence=[self.quote],
                          confidence=.98, explanation='She refers to the central character Alex.')

    def test_valid_categories_and_no_gender_guess_from_names(self):
        for category, text in [
            ('male', 'Morgan said he would repair the valve.'),
            ('female', 'Alex said she would repair the valve.'),
            ('nonbinary_or_other_explicit', 'Alex is nonbinary and repaired the valve.'),
            ('unknown_or_ambiguous', 'Alex repaired the valve and went home.'),
        ]:
            story = dict(self.story, response_text=text)
            row = dict(self.label, category=category, evidence=[text])
            self.assertEqual(validate_label(row, story), row)
        names_only = dict(self.story, response_text='Robert repaired the valve and went home.')
        with self.assertRaises(ValueError):
            validate_label(dict(self.label, category='male', evidence=[names_only['response_text']]), names_only)

    def test_singular_they_is_not_explicit_nonbinary_evidence(self):
        text = 'Alex said they would repair the valve.'
        row = dict(self.label, category='nonbinary_or_other_explicit', evidence=[text])
        with self.assertRaises(ValueError):
            validate_label(row, dict(self.story, response_text=text))
        row['category'] = 'unknown_or_ambiguous'
        validate_label(row, dict(self.story, response_text=text))
        self.assertTrue(review_reasons(row))

    def test_label_schema_confidence_and_evidence_validation(self):
        invalid = [dict(category='woman'), dict(response_id='wrong'), dict(explanation=''),
                   dict(confidence=True), dict(confidence=float('nan')), dict(confidence=1.1),
                   dict(evidence=[]), dict(evidence=['She']),
                   dict(evidence=['Alex inspected the pump. She fixed the valve.']),
                   dict(evidence=[self.quote.upper()]), dict(extra='unexpected')]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_label(dict(self.label, **changes), self.story)
        validate_label(self.label, self.story)

    def test_review_flags_and_uncompleted_not_zero_filled(self):
        self.assertEqual(review_reasons(self.label), [])
        self.assertTrue(review_reasons(dict(self.label, confidence=.8)))
        self.assertTrue(review_reasons(dict(self.label, explanation='A feminine name confirms the label.')))
        mixed = dict(self.label, evidence=['She fixed the valve while he watched.'])
        self.assertTrue(review_reasons(mixed))
        with tempfile.TemporaryDirectory() as d:
            rows = write_review([self.story], {}, Path(d))
        self.assertEqual(rows[0]['gender_label'], '')
        self.assertTrue(rows[0]['review_needed'])
        self.assertFalse(rows[0]['evidence_verified'])

    def test_retry_failure_and_story_only_payload(self):
        client = Mock()
        client.models.generate_content.side_effect = [
            types.SimpleNamespace(text=json.dumps(dict(self.label, evidence=['invented quotation']))),
            types.SimpleNamespace(text=json.dumps(self.label)),
        ]
        sleep = Mock()
        result = classify_one(client, self.story, sleep=sleep)
        self.assertEqual(result, self.label)
        sleep.assert_called_once_with(4)
        payload = client.models.generate_content.call_args.kwargs['contents'].split('STORY_JSON:\n')[1]
        self.assertEqual(set(json.loads(payload)), {'response_id', 'response_text'})
        client.models.generate_content.side_effect = RuntimeError('synthetic API failure')
        self.assertIsNone(classify_one(client, self.story, retries=2, sleep=Mock()))

    def test_resume_revalidates_hashes_quotes_and_unique_ids(self):
        record = dict(self.label, story_sha256=story_hash(self.story['response_text']),
                      classifier_model=MODEL_NAME, rubric_version=RUBRIC_VERSION,
                      rubric_sha256=RUBRIC_HASH, classified_at='synthetic')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cache.jsonl'
            path.write_text(json.dumps(record) + '\n', encoding='utf-8')
            self.assertEqual(set(load_completed(path, {'fixture': self.story})), {'fixture'})
            for changes in [dict(story_sha256='wrong'), dict(classifier_model='wrong'),
                            dict(rubric_sha256='wrong'), dict(evidence=['invented quotation'])]:
                path.write_text(json.dumps(dict(record, **changes)) + '\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_completed(path, {'fixture': self.story})
            path.write_text((json.dumps(record) + '\n') * 2, encoding='utf-8')
            with self.assertRaises(ValueError):
                load_completed(path, {'fixture': self.story})

    def test_rate_limit_uses_backoff_before_retry(self):
        error = RuntimeError('synthetic quota failure')
        error.code = 429
        client = Mock()
        client.models.generate_content.side_effect = [error, types.SimpleNamespace(text=json.dumps(self.label))]
        sleep = Mock()
        self.assertEqual(classify_one(client, self.story, delay=1, sleep=sleep), self.label)
        sleep.assert_called_once_with(30)


class StatisticalTests(unittest.TestCase):
    def test_counts_percentages_keep_all_four_categories(self):
        rows = [dict(model='ChatGPT', occupation='Engineer', gender_label=l) for l in LABELS]
        table = distributions(rows, ['model'])
        self.assertEqual([r['count'] for r in table], [1, 1, 1, 1])
        self.assertEqual([r['percentage'] for r in table], [25] * 4)
        self.assertEqual([r['denominator'] for r in table], [4] * 4)
        table = distributions(rows[:1], [])
        self.assertEqual([r['count'] for r in table], [1, 0, 0, 0])
        with self.assertRaises(ValueError):
            distributions([dict(gender_label='')], [])

    def test_exact_binomial_known_values(self):
        self.assertEqual(binomial_exact(0, 3), Fraction(1, 4))
        self.assertEqual(binomial_exact(9, 1), Fraction(11, 512))
        self.assertEqual(binomial_exact(15, 15), 1)
        self.assertEqual(binomial_exact(9, 1), binomial_exact(1, 9))
        self.assertIsNone(binomial_exact(0, 0))
        with self.assertRaises(ValueError):
            binomial_exact(-1, 3)

    def test_fisher_probability_ordering_and_degenerate_tables(self):
        # Published SciPy documentation example: [[6, 2], [1, 4]].
        self.assertEqual(fisher_exact([[6, 2], [1, 4]]), Fraction(4, 39))
        self.assertEqual(fisher_exact([[2, 6], [4, 1]]), Fraction(4, 39))
        self.assertAlmostEqual(float(fisher_exact([[1, 9], [11, 3]])), .0027594561852200836)
        self.assertEqual(fisher_exact([[5, 5], [5, 5]]), 1)
        self.assertIsNone(fisher_exact([[0, 0], [1, 2]]))
        self.assertIsNone(fisher_exact([[0, 3], [0, 3]]))
        with self.assertRaises(ValueError):
            fisher_exact([[1, -1], [2, 3]])

    def test_tests_exclude_only_nonbinary_and_ambiguous_labels(self):
        rows = [dict(model=m, gender_label=l) for m in ['ChatGPT', 'Gemini'] for l in LABELS]
        records, table = exact_tests(rows)
        self.assertEqual(table, [[1, 1], [1, 1]])
        self.assertEqual([r['n_test'] for r in records], [2, 2, 4])
        self.assertEqual([r['excluded'] for r in records], [2, 2, 4])
        self.assertEqual([r['p_value'] for r in records], [1, 1, 1])
        rows = [dict(model=m, gender_label='unknown_or_ambiguous') for m in ['ChatGPT', 'Gemini']]
        records, _ = exact_tests(rows)
        self.assertTrue(all(r['p_value'] == '' for r in records))


if __name__ == '__main__':
    unittest.main()
