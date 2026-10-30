"""Generate Gemini stories from the prompt workbook, resuming completed responses."""

import os
import csv
import time
import openpyxl
from google import genai
from common.paths import ROOT, DATA
from dotenv import load_dotenv
from common.data_utils import RESPONSE_COLUMNS

load_dotenv(ROOT / '.env')


EXCEL_PATH = DATA / "prompt_dataset.xlsx"
OUTPUT_CSV = DATA / "responses_gemini.csv"
MODEL_NAME = "gemini-flash-lite-latest"
REPETITIONS = 3
DELAY_SECONDS = 4
MAX_RETRIES = 3


client = None

CSV_HEADERS = RESPONSE_COLUMNS


def load_prompts():
    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb["Prompts"]
    prompts = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        prompt_id, occupation, scenario, gender_version, prompt_text = row[:5]
        if prompt_id is None:
            continue
        prompts.append({
            "prompt_id": prompt_id,
            "occupation": occupation,
            "gender_version": gender_version,
            "prompt_text": prompt_text,
        })
    wb.close()
    return prompts


def load_existing_keys():
    """Return completed prompt/repetition keys, excluding failed responses."""
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["response_text"].startswith("[ERROR:"):
                    continue
                done.add((row["prompt_id"], int(row["repetition"])))
    return done


def ensure_csv_header():
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
            if next(csv.reader(f), []) != CSV_HEADERS:
                raise ValueError('Existing output schema differs; refusing to append incompatible rows')
    else:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def call_gemini(prompt_text):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_text,
            )
            return response.text
        except Exception as e:
            last_error = e
            print(f"  attempt {attempt} failed: {safe_error(e)}")
            time.sleep(DELAY_SECONDS * attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {safe_error(last_error)}") from None


def safe_error(error):
    """Preserve useful error details while redacting the configured credential."""
    message = str(error)
    key = os.environ.get("GEMINI_API_KEY")
    return message.replace(key, "[REDACTED]") if key else message


def main():
    global client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY environment variable not set.")
    client = genai.Client(api_key=api_key)
    prompts = load_prompts()
    done_keys = load_existing_keys()
    ensure_csv_header()

    total_calls = len(prompts) * REPETITIONS
    call_num = 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for p in prompts:
            for rep in range(1, REPETITIONS + 1):
                call_num += 1
                key = (p["prompt_id"], rep)
                if key in done_keys:
                    print(f"[{call_num}/{total_calls}] SKIP {p['prompt_id']} rep {rep} (already done)")
                    continue

                print(f"[{call_num}/{total_calls}] Generating {p['prompt_id']} "
                      f"({p['occupation']}, {p['gender_version']}) rep {rep} ...")

                try:
                    story_text = call_gemini(p["prompt_text"])
                except Exception as e:
                    print(f"  GAVE UP: {e}")
                    story_text = f"[ERROR: {e}]"

                response_id = f"R_{p['prompt_id']}_gem_{rep}"
                writer.writerow([
                    response_id, p["prompt_id"], p["occupation"], p["gender_version"],
                    "Gemini", rep, story_text,
                    "", "", "", "", "", "",
                ])
                f.flush()
                time.sleep(DELAY_SECONDS)

    print(f"\nDone. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
