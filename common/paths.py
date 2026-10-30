"""Project locations, anchored to this file rather than the terminal directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARCHIVE = DATA / 'archive'
RESULTS = ROOT / 'results'
RAW_JUDGE = ARCHIVE / 'before_llm_transfer' / 'agency_judge_scores.csv'
CLEAN_JUDGE = DATA / 'agency_judge_scores_clean.csv'
MANUAL = DATA / 'human_reviewed_stories.xlsx'
