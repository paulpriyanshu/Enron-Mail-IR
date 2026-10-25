"""Run once: read emails.csv -> parse -> build TF-IDF + BM25 indexes -> save to index.pkl."""
import csv
import pickle
import time

from search_engine import build_bm25_index, build_tfidf_index, parse_email

CSV_PATHS = ["fake_emails.csv", "emails.csv"]  # fake spam first (so LIMIT keeps it), then Kaggle Enron (columns: file, message)
LIMIT = None  # None = all emails; set e.g. 50_000 for a faster, smaller index

csv.field_size_limit(2**31 - 1)  # some emails are bigger than csv's default limit

start = time.time()
emails = []
seen = set()  # Enron keeps the same email in several folders (sent, all_documents...)
for path in CSV_PATHS:
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            e = parse_email(row["message"])
            key = (e["from"], e["subject"], e["body"])
            if key in seen:
                continue  # skip duplicates
            seen.add(key)
            emails.append(e)
            if len(emails) == LIMIT:
                break
print(f"Parsed {len(emails)} unique emails")

tfidf_index = build_tfidf_index(emails)
bm25_index = build_bm25_index(emails)
print(f"Vocabulary size: {len(tfidf_index[0].vocabulary_)} words")

with open("index.pkl", "wb") as f:
    pickle.dump((emails, tfidf_index, bm25_index), f)
print(f"Saved index.pkl in {time.time() - start:.1f}s")
