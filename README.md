# Enron Email Search — TF-IDF vs BIM vs BM25

Search ~250k unique Enron emails, with TF-IDF, BIM and BM25 results side by side.

## Setup (once)
```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download `emails.csv` from https://www.kaggle.com/datasets/wcukierski/enron-email-dataset into this folder.

## Run
```bash
source .venv/bin/activate         # once per terminal (Windows: .venv\Scripts\activate)
python test_search.py             # check everything works
python compare.py                 # small example: each method picks a different top email
python load_data.py               # build index.pkl (~75s, only needed once)
python app.py                     # open http://localhost:8000
```

## The three ranking algorithms
| | Uses | Weakness |
|---|---|---|
| **TF-IDF** + cosine | word counts × rarity, normalised by length | favours tiny emails |
| **BIM** (Binary Independence Model) | only *present or absent* × rarity (IDF) | hundreds of emails tie for #1 |
| **BM25** (k1=0.5, b=0.75) | BIM + repeat counting (saturates) + length normalisation | — the one search engines use |

All three are unsupervised (no training data), so they work from the first query.
BIM and BM25 share one index; all three find the same emails, only the order differs.

Good demo queries: `stock options`, `sec investigation`, `power outage`, `bankruptcy`.
More detail: `PROJECT_EXPLAINED.md` (concepts, worked examples, viva Q&A) and `CODE_EXPLAINED.md` (line by line).

## Files
- `search_engine.py` — parsing, TF-IDF, BIM and BM25 search
- `load_data.py` — reads emails.csv, builds and saves the index
- `app.py` + `templates/index.html` — the web app
- `compare.py` — TF-IDF vs BIM vs BM25 demo
- `test_search.py` — tests
