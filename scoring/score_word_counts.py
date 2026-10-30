"""Count competence and uncertainty word roots in response CSVs."""
import sys
import csv


COMPETENCE_ROOTS = [
    'skill', 'decisive', 'capab', 'clever', 'brilliant',
    'confiden', 'expert', 'competen', 'resourceful', 'determin',
    'intelligen', 'compos', 'calm', 'focus', 'precis',
    'efficient', 'masterful', 'authorit', 'commanding', 'assur',
    'steady', 'strategic', 'sharp', 'adept', 'proficien',
    'leader', 'control', 'innovat', 'bold', 'fearless',
]

UNCERTAINTY_ROOTS = [
    'hesitant', 'nervous', 'uncertain', 'anxious', 'unsure',
    'worried', 'doubt', 'overwhelm', 'fearful', 'timid',
    'insecure', 'shaky', 'trembl', 'panick', 'frightened',
    'scared', 'afraid', 'confus', 'helpless', 'vulnerable',
    'fragile', 'meek', 'reluctan', 'apprehensive', 'tearful',
    'sob', 'crying', 'weak', 'distraught', 'flustered',
]


def tokenize(text):
    """Original tokenisation: replace only these eight ASCII punctuation marks."""
    text = text.lower()
    for punctuation in ",.\"';:!?":
        text = text.replace(punctuation, " ")
    return text.split()


def count_root_occurrences(text, roots):
    if not text:
        return 0

    words = tokenize(text)

    total_count = 0
    for root in roots:
        for word in words:
            if word.startswith(root):
                total_count += 1

    return total_count


def read_csv_robust(filename):
    """Read UTF-8 CSVs, with a Windows text-encoding fallback."""
    for encoding in ["utf-8-sig", "utf-8", "cp1252"]:
        try:
            with open(filename, newline="", encoding=encoding) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)
            print(f"(read {filename} using {encoding} encoding)")
            return fieldnames, rows
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Could not read {filename} with any known encoding.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scoring.score_word_counts <csv_filename>")
        print("Example: python -m scoring.score_word_counts data/responses_gemini.csv")
        sys.exit(1)

    filename = sys.argv[1]

    fieldnames, rows = read_csv_robust(filename)

    if "response_text" not in fieldnames:
        raise SystemExit(f"'response_text' column not found in {filename}")

    scored_count = 0
    for row in rows:
        text = row.get("response_text", "")

        if not text or text.strip() == "" or text.strip().startswith("[ERROR:"):
            continue

        competence_count = count_root_occurrences(text, COMPETENCE_ROOTS)
        uncertainty_count = count_root_occurrences(text, UNCERTAINTY_ROOTS)

        row["competence_word_count"] = competence_count
        row["emotional_support_word_count"] = uncertainty_count
        scored_count += 1

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Scored {scored_count} responses in {filename}.")
    print("competence_word_count and emotional_support_word_count columns "
          "have been filled in for every story with response text.")


if __name__ == "__main__":
    main()
