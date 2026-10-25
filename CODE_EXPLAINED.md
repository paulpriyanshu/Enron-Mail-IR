# Code Explained — line by line

This guide teaches every file of the project. All example outputs below are **real outputs** from running the code.

---

## 0. Who wrote what? (library vs our code)

| Piece | Written by | Where it comes from |
|---|---|---|
| `email.message_from_string` | Python | built-in `email` module |
| `csv.DictReader` | Python | built-in `csv` module |
| `pickle.dump` / `pickle.load` | Python | built-in `pickle` module |
| **`CountVectorizer`** | **scikit-learn** | `sklearn.feature_extraction.text` |
| **`TfidfVectorizer`** | **scikit-learn** | `sklearn.feature_extraction.text` |
| `.fit_transform()`, `.transform()`, `.build_analyzer()`, `.vocabulary_` | scikit-learn | methods of the two vectorizers |
| Sparse matrices (`csr_matrix`, `.tocsc()`, `.indices`, `.data`) | SciPy | returned by the vectorizers |
| `np.zeros`, `np.log`, `.nonzero()`, `.argsort()` | NumPy | math on arrays |
| `Flask`, `@app.route`, `render_template`, `request` | Flask | web framework |
| `escape`, `Markup` | MarkupSafe | installed with Flask |
| **`parse_email()`** | **us** | `search_engine.py` |
| **`email_texts()`, `rank()`** | **us** | `search_engine.py` |
| **`build_tfidf_index()`, `search_tfidf()`** | **us** (using scikit-learn inside) | `search_engine.py` |
| **`build_bm25_index()`** | **us** (using `CountVectorizer` for counting) | `search_engine.py` |
| **`search_bm25()` — the BM25 formula** | **us, entirely** | `search_engine.py` |
| **Duplicate removal, loading, saving** | **us** | `load_data.py` |
| **`highlight()`, routes, web page** | **us** | `app.py`, `templates/index.html` |

**In one sentence:** scikit-learn does the *text → numbers* step (tokenising, counting, TF-IDF weights); **we** wrote the search logic — cosine scoring, **the whole BM25 formula**, ranking, data loading and the web app.

---

## 1. The library pieces, explained

### 1.1 `CountVectorizer` (scikit-learn) — text → word counts

It does 3 jobs:

```
"The Gas PIPELINE is down!"
   │ 1. ANALYZER: lowercase → split into words of 2+ letters → remove stop words
   ▼
['gas', 'pipeline']                                         ← real output of build_analyzer()
   │ 2. VOCABULARY: give every unique word a column number (learned by fit)
   ▼
{'california': 0, 'friday': 1, 'gas': 2, 'lunch': 3, 'pipeline': 4, ...}
   │ 3. COUNT: one row per email, one column per word (a sparse matrix)
   ▼
[[1 0 2 0 0 0 2 1 0]      ← Email 0: california×1, gas×2, prices×2, rising×1
 [0 0 1 0 2 0 0 0 1]      ← Email 1: gas×1, pipeline×2, shut×1
 [0 1 0 1 0 1 0 0 0]]     ← Email 2: friday×1, lunch×1, pizza×1
```

| Method / attribute | What it does |
|---|---|
| `fit_transform(texts)` | **Learn** the vocabulary from the texts **and** return the count matrix |
| `transform(texts)` | Use the **already learned** vocabulary to turn new text (a query) into a row |
| `build_analyzer()` | Returns the tokenising function (step 1 alone) |
| `vocabulary_` | The dict `word → column number` |
| `stop_words="english"` | Use scikit-learn's built-in list of ~318 English stop words |

**Could we write it ourselves?** Yes — the core idea is a few lines of plain Python:
```python
import re
from collections import Counter

def analyze(text):
    return [w for w in re.findall(r"\b\w\w+\b", text.lower()) if w not in STOP_WORDS]

vocabulary = {}
rows = []
for text in texts:
    counts = Counter(analyze(text))                 # {"gas": 2, "prices": 2, ...}
    for word in counts:
        vocabulary.setdefault(word, len(vocabulary))  # give new words the next column number
    rows.append(counts)
```
We used scikit-learn because it is standard, tested, and runs in fast C code (a pure-Python loop over 251k emails is much slower).

### 1.2 `TfidfVectorizer` (scikit-learn) — text → TF-IDF vectors

Same as `CountVectorizer`, then two more steps:

```
counts  ──►  × IDF  ──►  L2-normalise each row (length = 1)
```
- **IDF** (smoothed): `ln((1 + N) / (1 + df)) + 1`
- **L2 normalise**: divide each row by its length `√(x₁² + x₂² + …)`, so every vector has length 1

Real output for our 3 emails (`matrix.toarray()`):
```
Email 0: [0.35 0.   0.53 0.   0.   0.   0.69 0.35 0.  ]
Email 1: [0.   0.   0.32 0.   0.85 0.   0.   0.   0.42]
Email 2: [0.   0.58 0.   0.58 0.   0.58 0.   0.   0.  ]
```
Query `"gas pipeline"` → `vectorizer.transform([...])` → `[0. 0. 0.61 0. 0.8 0. 0. 0. 0.]`

### 1.3 SciPy sparse matrices

The vectorizers return a **sparse matrix** (only non-zero values are stored — see `PROJECT_EXPLAINED.md` §6).

| Code | Meaning |
|---|---|
| `matrix` (CSR) | Stored **row by row** (email by email) |
| `.tocsc()` | Convert to **CSC** — stored **column by column** (word by word) = inverted index |
| `counts[:, col]` | Get one column (one word) |
| `column.indices` | **Which rows** (emails) have a non-zero value in that column |
| `column.data` | **The values** (the counts) |
| `counts.sum(axis=1)` | Sum each row → total words in each email |
| `.A1` | Turn a 1-column matrix into a flat NumPy array |
| `.toarray()` | Turn sparse into a normal (dense) array — only safe when it's small |
| `A @ B` | Matrix multiplication |
| `.T` | Transpose (flip rows and columns) |

Real example: `counts[:, vocabulary_["gas"]]` → `indices = [0 1]`, `data = [2 1]` → "gas is in Email 0 (2 times) and Email 1 (1 time)".

### 1.4 NumPy functions we use

| Code | Meaning | Example |
|---|---|---|
| `np.zeros(n)` | Array of `n` zeros | `[0. 0. 0.]` |
| `np.log(x)` | Natural log (ln) | `np.log(2.718) ≈ 1` |
| `a.nonzero()[0]` | Positions where the value isn't 0 | `[0, .869, 0, .319]` → `[1 3]` |
| `a.argsort()` | Positions that would sort the array (low → high) | |
| `[::-1]` | Reverse (→ high → low) | |
| `a.mean()` | Average | |
| `scores[rows] += x` | Add to many positions at once ("fancy indexing") | |

---

## 2. `search_engine.py` — the core (our code)

### Imports (lines 2–5)
```python
import email                                                           # Python: parse raw emails
import numpy as np                                                     # NumPy: fast array math
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer   # scikit-learn
```

### `parse_email(raw)` (lines 8–19) — raw text → dict
```python
def parse_email(raw):
    msg = email.message_from_string(raw)     # Python parses headers + body
    if msg.is_multipart():                   # some emails have attachments/parts
        msg = msg.get_payload(0)             # keep only the first part (the text)
    return {
        "from": msg.get("From", ""),         # header value, or "" if missing
        "to": msg.get("To", ""),
        "date": msg.get("Date", ""),
        "subject": msg.get("Subject", ""),
        "body": msg.get_payload(),           # everything after the headers
    }
```
Real output:
```python
parse_email("From: ken@enron.com\nSubject: Hi\n\nBody text")
# {'from': 'ken@enron.com', 'to': '', 'date': '', 'subject': 'Hi', 'body': 'Body text'}
```

### `email_texts(emails)` (lines 22–23) — what we actually search
```python
def email_texts(emails):
    return [e["subject"] + " " + e["body"] for e in emails]
```
A **list comprehension**: for every email, join subject + body into one string. So words in the subject are searchable too.

### `rank(scores)` (lines 26–30) — scores → sorted results
```python
def rank(scores):
    matches = scores.nonzero()[0]                       # 1. positions with score > 0
    ranked = matches[scores[matches].argsort()[::-1]]   # 2. sort them by score, high → low
    return [(int(i), float(scores[i])) for i in ranked] # 3. list of (email_id, score)
```
Step by step with `scores = [0, 0.869, 0, 0.319]`:
```
1. matches                 = [1, 3]              (emails 0 and 2 scored 0 → dropped)
2. scores[matches]         = [0.869, 0.319]
   .argsort()              = [1, 0]              (positions in low → high order)
   [::-1]                  = [0, 1]              (high → low)
   matches[[0, 1]]         = [1, 3]
3. result                  = [(1, 0.869), (3, 0.319)]     ← real output
```
Both algorithms share this function — only the scores differ.

### `build_tfidf_index(emails)` (lines 35–39)
```python
def build_tfidf_index(emails):
    vectorizer = TfidfVectorizer(stop_words="english")       # scikit-learn object, not trained yet
    matrix = vectorizer.fit_transform(email_texts(emails))   # learn vocabulary + IDF, build vectors
    return vectorizer, matrix                                # keep the vectorizer for queries later
```
We return the **vectorizer** too, because the query must be turned into a vector **with the same vocabulary and IDF values**.

### `search_tfidf(query, index)` (lines 42–47)
```python
def search_tfidf(query, index):
    vectorizer, matrix = index                               # unpack the tuple
    query_vector = vectorizer.transform([query])             # query → TF-IDF vector (1 row)
    scores = (matrix @ query_vector.T).toarray().ravel()     # cosine with EVERY email at once
    return rank(scores)
```
The key line, piece by piece:
```
matrix               251,115 × 631,253   (one row per email)
query_vector.T       631,253 × 1         (query as a column)
matrix @ query.T     251,115 × 1         (one score per email = dot product)
.toarray()           sparse → normal array
.ravel()             flatten to 1-D: [0.319, 0.869, 0.0]      ← real output for 3 emails
```
**Why is a dot product = cosine similarity?** `cos = (q · e) / (|q| × |e|)`, and `TfidfVectorizer` already made every vector length 1, so `|q| × |e| = 1` → `cos = q · e`.

### `build_bm25_index(emails)` (lines 52–57)
```python
def build_bm25_index(emails):
    counter = CountVectorizer(stop_words="english")                 # scikit-learn: just counting
    counts = counter.fit_transform(email_texts(emails)).tocsc()     # count matrix → CSC (by word)
    doc_len = counts.sum(axis=1).A1                                 # words per email: [6 4 3]
    return counter, counts, doc_len
```
- BM25 needs **raw counts** (not TF-IDF weights), so we use `CountVectorizer`.
- `.tocsc()` makes "which emails contain word X?" instant — the inverted index.
- `doc_len` is needed for length normalisation.

### `search_bm25(query, index, k1=1.5, b=0.75)` (lines 60–83) — **100% our code**
```python
def search_bm25(query, index, k1=1.5, b=0.75):
    counter, counts, doc_len = index
    n_emails = counts.shape[0]            # N = number of emails (rows)
    avg_len = doc_len.mean()              # average email length
    scores = np.zeros(n_emails)           # start every email at score 0

    for word in counter.build_analyzer()(query):        # tokenise the query exactly like the emails
        col = counter.vocabulary_.get(word)             # which column is this word?
        if col is None:
            continue                                    # word not in any email → skip
        column = counts[:, col]                         # the word's column (CSC = fast)
        rows, tf = column.indices, column.data          # emails containing it + how often
        df = len(rows)                                  # document frequency
        idf = np.log((n_emails - df + 0.5) / (df + 0.5) + 1)                      # BM25 IDF
        scores[rows] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[rows] / avg_len))

    return rank(scores)
```
How the last formula line maps to the BM25 formula:
```
                    IDF   ×  tf  × (k1 + 1)
score += ─────────────────────────────────────────────────────
           tf + k1 × (1 − b + b × doc_len / avg_len)
                     └──── length normalisation ────┘
```
Walk-through for `"gas pipeline"` on our 3 emails:
```
word "gas":      rows = [0, 1], tf = [2, 1], df = 2, idf = 0.470
                 scores[[0, 1]] += [0.598, 0.487]    → scores = [0.598, 0.487, 0]
word "pipeline": rows = [1],    tf = [2],    df = 1, idf = 0.981
                 scores[[1]]    += [1.437]           → scores = [0.598, 1.924, 0]
rank(scores)  → [(1, 1.924), (0, 0.598)]
```
**Why it's fast:** each word only updates the emails in `rows` — Email 2 is never touched.
The calculation runs on whole arrays at once (`tf`, `doc_len[rows]` are arrays) — this is called **vectorised** NumPy code, much faster than a Python loop over emails.

---

## 3. `load_data.py` — build the index (run once)

```python
import csv, pickle, time
from search_engine import build_bm25_index, build_tfidf_index, parse_email

CSV_PATH = "emails.csv"
LIMIT = None                         # None = all emails; e.g. 50_000 for a quick test

csv.field_size_limit(2**31 - 1)      # default limit is 131,072 characters; some emails are bigger
```

```python
start = time.time()
emails = []
seen = set()                                          # fast "have I seen this?" lookups
with open(CSV_PATH, encoding="utf-8", errors="replace") as f:   # replace broken characters
    for row in csv.DictReader(f):                     # each row = {"file": ..., "message": ...}
        e = parse_email(row["message"])
        key = (e["from"], e["subject"], e["body"])    # identity of an email
        if key in seen:
            continue                                  # duplicate → skip
        seen.add(key)
        emails.append(e)
        if len(emails) == LIMIT:                      # never true when LIMIT is None
            break
```
- `with open(...) as f:` — opens the file and **closes it automatically** at the end.
- `csv.DictReader` — reads one row at a time (doesn't load 1.4 GB into memory at once).
- A `set` checks membership in constant time — fast even with 251k emails.

```python
tfidf_index = build_tfidf_index(emails)      # (vectorizer, matrix)
bm25_index = build_bm25_index(emails)        # (counter, counts, doc_len)

with open("index.pkl", "wb") as f:
    pickle.dump((emails, tfidf_index, bm25_index), f)    # save everything in one file
```

---

## 4. `app.py` — the web app

### Setup (lines 11–14)
```python
app = Flask(__name__)                        # create the web app

with open("index.pkl", "rb") as f:           # runs ONCE when the app starts
    emails, tfidf_index, bm25_index = pickle.load(f)
```

### `highlight(text, query)` (lines 17–24) — wrap query words in `<mark>`
```python
def highlight(text, query):
    text = str(escape(text))                                    # 1. make the email safe for HTML
    words = [re.escape(str(escape(w))) for w in query.split()]  # 2. make query words safe for regex
    if words:
        pattern = r"\b(" + "|".join(words) + r")\b"             # 3. e.g. \b(gas|pipeline)\b
        text = re.sub(pattern, r"<mark>\1</mark>", text, flags=re.IGNORECASE)   # 4. wrap matches
    return Markup(text)                                         # 5. tell Flask "this HTML is safe"
```
Real output: `highlight("Gas <prices> rising", "gas")` → `<mark>Gas</mark> &lt;prices&gt; rising`

- **`escape`** turns `<` into `&lt;` so an email containing HTML/JavaScript can't run in the browser (prevents **XSS attacks**).
- **`re.escape`** stops characters like `.` or `*` in the query from acting as regex symbols.
- **`\b`** = word boundary → "gas" doesn't highlight inside "Vegas".
- **`\1`** = "the text that matched", so the original capitalisation is kept.

```python
app.jinja_env.filters["highlight"] = highlight   # lets the HTML template use {{ text | highlight(query) }}
```

### `timed_search` (lines 30–33)
```python
def timed_search(search, query, index):          # `search` is a function passed in as a value
    start = time.time()
    results = search(query, index)
    return {"results": results[:10], "found": len(results), "took": time.time() - start}
```

### Routes (lines 36–52) — which URL runs which function
```python
@app.route("/")                                  # http://localhost:8000/?q=gas+pipeline
def home():
    query = request.args.get("q", "").strip()    # read ?q=... from the URL ("" if missing)
    columns = {}
    if query:
        columns = {
            "TF-IDF": timed_search(search_tfidf, query, tfidf_index),
            "BM25": timed_search(search_bm25, query, bm25_index),
        }
    return render_template("index.html", query=query, columns=columns, emails=emails)

@app.route("/email/<int:i>")                     # http://localhost:8000/email/42
def show_email(i):                               # Flask converts "42" → the int 42
    if i >= len(emails):
        abort(404)                               # no such email → "Not Found" page
    return render_template("index.html", query=request.args.get("q", ""), email=emails[i], emails=emails)

if __name__ == "__main__":                       # only when run as `python app.py`
    app.run(debug=True, port=8000)               # 5000 is taken by macOS AirPlay
```
- **`@app.route`** is a *decorator*: it registers the function for that URL.
- **`render_template`** fills `templates/index.html` with the given variables and returns HTML.

---

## 5. `templates/index.html` — the page (Jinja2 template)

Jinja2 is HTML with placeholders that Flask fills in:

| Syntax | Meaning |
|---|---|
| `{{ x }}` | Print a value (auto-escaped for safety) |
| `{{ x \| highlight(query) }}` | Pass `x` through a filter (our `highlight` function) |
| `{% if %} … {% elif %} … {% endif %}` | Conditions |
| `{% for a, b in items %} … {% else %} … {% endfor %}` | Loop; `else` runs if the list is empty |
| `{% set e = emails[i] %}` | Create a variable |
| `{{ loop.index }}` | 1, 2, 3… inside a loop |
| `{# … #}` | Comment |

The page has 3 states:
```
{% if email %}        → single email view (from /email/<id>)
{% elif query %}      → two columns of results: for name, col in columns.items()
(nothing)             → just the search box
```
The `<style>` block at the top is plain CSS: `.columns` uses CSS **grid** for two columns side by side,
and a `@media (max-width: 700px)` rule stacks them on phones.

---

## 6. `compare.py` and `test_search.py`

**`compare.py`** — builds both indexes on 5 hand-made emails and prints both rankings:
```python
if __name__ == "__main__":       # runs only with `python compare.py`, not when imported
    show("TF-IDF ...", search_tfidf(QUERY, build_tfidf_index(EMAILS)))
    show("BM25:",      search_bm25(QUERY, build_bm25_index(EMAILS)))
```
The saturation table uses the BM25 TF part at average length: `tf * 2.5 / (tf + 1.5)` = `tf × (k1+1) / (tf + k1)`.

**`test_search.py`** — plain `assert` statements; if any is false, Python stops with an error:
```python
assert [i for i, _ in search("gas", index)] == [2, 0]   # most "gas" first, pizza not matched
assert search("banana", index) == []                     # unknown word → no results
assert search_tfidf(QUERY, ...)[0][0] == 0               # TF-IDF puts the junk email first
assert search_bm25(QUERY, ...)[0][0] == 1                # BM25 puts the real email first
```
The loop `for build, search in [(build_tfidf_index, search_tfidf), (build_bm25_index, search_bm25)]`
runs the same checks on both algorithms — functions can be stored in lists like any other value.

---

## 7. Viva answers

**Q: Did you implement CountVectorizer / TfidfVectorizer?**
No — they're from scikit-learn. They do tokenising, stop-word removal, building the vocabulary and counting / TF-IDF weighting.
We used them because they're standard and fast. The same logic can be written in a few lines with `Counter` (section 1.1).

**Q: Then what did you implement?**
Email parsing and duplicate removal, the search functions, the **complete BM25 formula** (IDF, saturation with k1,
length normalisation with b, scoring via the inverted index), the shared ranking function, saving/loading the index,
and the web app with safe highlighting.

**Q: Why CountVectorizer for BM25 and TfidfVectorizer for TF-IDF?**
BM25 needs **raw counts** and its own IDF formula. `TfidfVectorizer` already applies scikit-learn's IDF and normalisation,
which would be wrong for BM25.

**Q: What's the difference between `fit_transform` and `transform`?**
`fit_transform` **learns** the vocabulary (and IDF) from the emails and converts them. `transform` **reuses** what was learned
to convert a query — so the query's columns line up with the emails' columns.

**Q: Why `.tocsc()`?**
BM25 looks up one word at a time. CSC stores the matrix column by column, so getting "all emails containing this word" is instant — it's an inverted index.
