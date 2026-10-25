# Enron Email Search — TF-IDF vs BM25

Search ~250k unique Enron emails, with TF-IDF and BM25 results side by side.

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
python compare.py                 # small example: why BM25 beats TF-IDF
python load_data.py               # build index.pkl (~75s, only needed once)
python app.py                     # open http://localhost:8000
```

## Files
- `search_engine.py` — parsing, TF-IDF search, BM25 search
- `load_data.py` — reads emails.csv, builds and saves the index
- `app.py` + `templates/index.html` — the web app
- `compare.py` — TF-IDF vs BM25 demo
- `test_search.py` — tests
