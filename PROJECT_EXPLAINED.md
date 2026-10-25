# Enron Email Search — TF-IDF vs BIM vs BM25

## 1. What is this project?

A search engine (like Gmail's search bar) over the **Enron email dataset**. You type a query such as
`stock options`, and it returns the most relevant emails — ranked by **three algorithms side by side**:
**TF-IDF**, **BIM** and **BM25** — so we can compare how well each one ranks.

**Goal:** understand how text search engines work internally, implement three classic ranking
algorithms, and show with real data why **BM25** (used by Elasticsearch, Lucene, Solr) is better than
plain TF-IDF and than its ancestor, the Binary Independence Model (BIM).

---

## 2. The dataset

- In 2001 **Enron**, a huge US energy company, collapsed after a massive accounting fraud.
- During the investigation, the US regulator (**FERC**) made the employees' emails public.
- It is the **largest public collection of real corporate email** — widely used for research in search, NLP and spam detection.

| Fact | Value |
|---|---|
| Source | Kaggle "Enron Email Dataset" (`emails.csv`, 1.4 GB) |
| Total emails | **517,401** |
| Mailboxes (employees) | **150** |
| Unique emails after removing duplicates | **251,115** |
| Unique words (vocabulary) | **631,253** |

**Why duplicates?** Enron's mail system stored the same email in several folders (`sent`, `_sent_mail`,
`all_documents`...). About half the dataset is copies, so we skip any email whose sender + subject + body we already saw.

**One row of `emails.csv` looks like this:**
```
file:    allen-p/_sent_mail/1.
message: Message-ID: <18782981.1075855378110.JavaMail.evans@thyme>
         Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)
         From: phillip.allen@enron.com
         To: tim.belden@enron.com
         Subject:
         ...more headers...

         Here is our forecast
```

---

## 3. The whole data flow

The project has **two phases**: **indexing** (done once, offline) and **searching** (every time a user searches).

### Phase A — Indexing (`python load_data.py`, ~75 seconds, run once)

```
┌──────────────────────────┐
│ emails.csv (1.4 GB)      │  517,401 rows: file, message (raw email text)
└────────────┬─────────────┘
             │ csv.DictReader reads one row at a time        (load_data.py)
             ▼
┌──────────────────────────┐
│ 1. PARSE                 │  parse_email(raw)                (search_engine.py)
│   raw text → dict        │  Python's email module splits headers and body:
│                          │  {"from": "phillip.allen@enron.com",
│                          │   "to": "tim.belden@enron.com",
│                          │   "date": "Mon, 14 May 2001 ...",
│                          │   "subject": "", "body": "Here is our forecast"}
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 2. REMOVE DUPLICATES     │  key = (from, subject, body)
│   517,401 → 251,115      │  if key already seen → skip
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 3. PREPARE TEXT          │  email_texts(): subject + " " + body
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 4. TOKENISE              │  "Gas prices rising in California"
│   (inside scikit-learn)  │   → lowercase → split into words
│                          │   → remove stop words ("in", "the", "is"...)
│                          │   → ["gas", "prices", "rising", "california"]
└──────┬────────────┬──────┘
       │            │
       ▼            ▼
┌─────────────┐ ┌──────────────────┐
│ 5a. TF-IDF  │ │ 5b. BM25 + BIM   │
│ Tfidf-      │ │ CountVectorizer  │
│ Vectorizer  │ │ → word counts    │
│ → TF-IDF    │ │ + email lengths  │
│ vectors     │ │                  │
│ (CSR sparse │ │ (CSC sparse      │
│  matrix)    │ │  matrix, shared) │
└──────┬──────┘ └────────┬─────────┘
       └────────┬────────┘
                ▼
┌──────────────────────────┐
│ 6. SAVE WITH PICKLE      │  pickle.dump((emails, tfidf_index, bm25_index))
│   → index.pkl (1.1 GB)   │
└──────────────────────────┘
```

### Phase B — Searching (`python app.py`, then every search in the browser)

```
 App startup (once, ~1.3 s):  index.pkl ──pickle.load──► emails + matrices in RAM

┌───────────────┐  types "gas pipeline", presses Search
│ Browser       │ ─────────────────────────────────────────┐
└───────────────┘   GET http://localhost:8000/?q=gas+pipeline
        ▲                                                  ▼
        │                                  ┌──────────────────────────────┐
        │                                  │ Flask route home()  (app.py) │
        │                                  │ query = "gas pipeline"       │
        │                                  └─┬────────────┬─────────────┬─┘
        │                                    ▼            ▼             ▼
        │                          ┌──────────────┐┌──────────────┐┌──────────────┐
        │                          │search_tfidf()││search_bim()  ││search_bm25() │
        │                          │query → vector││emails with   ││emails with   │
        │                          │cosine with   ││each word:    ││each word:    │
        │                          │ALL emails    ││  + IDF       ││ + BM25 score │
        │                          └──────┬───────┘└──────┬───────┘└──────┬───────┘
        │                                 ▼               ▼               ▼
        │                          ┌──────────────────────────────────────────────┐
        │                          │ rank(): keep scores > 0, sort high → low     │
        │                          │ → [(email_id, score), ...]  take top 10      │
        │                          └───────────────────────┬──────────────────────┘
        │                                                  ▼
        │                          ┌──────────────────────────────────────────────┐
        │                          │ templates/index.html (Jinja2)                │
        │   HTML page              │ three columns: TF-IDF | BIM | BM25           │
        └───────────────────────── │ highlight() wraps query words in <mark>      │
                                   └──────────────────────────────────────────────┘

Click a result → GET /email/<id> → show_email() → full email page with highlights
```

---

## 4. Tech stack

| Layer | Technology | Why |
|---|---|---|
| Language | **Python 3** | Simple, best ecosystem for text/ML |
| Parsing | Python built-in `email`, `csv` | No extra install needed |
| TF-IDF | **scikit-learn** `TfidfVectorizer` | Standard, well-tested implementation |
| BIM | **Written by hand**, reuses the BM25 count matrix | Classic probabilistic baseline — shows what BM25 adds |
| BM25 | **Written by hand** with NumPy + scikit-learn `CountVectorizer` | To show we understand the formula |
| Math | **NumPy**, **SciPy** sparse matrices | Fast vector math, memory efficient |
| Storage | `pickle` → `index.pkl` | Build index once, load in ~1 second |
| Web backend | **Flask** | Tiny web framework, ~50 lines |
| Frontend | HTML + CSS (Jinja2 templates) | No build step, easy to read |
| Testing | `test_search.py` (plain asserts) | Checks all three algorithms rank correctly |

### Project files
| File | What it does |
|---|---|
| `search_engine.py` | Core logic: parse email, build TF-IDF + BM25 indexes, search with TF-IDF, BIM, BM25 |
| `load_data.py` | Reads `emails.csv`, removes duplicates, builds and saves the index |
| `app.py` | Flask web app (search page + single email page + word highlighting) |
| `templates/index.html` | The web page |
| `compare.py` | Hand-made example where each algorithm puts a different email first |
| `test_search.py` | Tests |

---

## 5. Worked example (used in sections 6–9)

To see what happens inside, take **3 tiny emails** and the query **`gas pipeline`**.
All numbers below were produced by the project's real code.

| Email | Subject + body | After tokenising (stop words removed) |
|---|---|---|
| 0 | "Gas prices" + "gas prices rising in california" | gas, prices, gas, prices, rising, california |
| 1 | "Pipeline" + "the gas pipeline is shut down" | pipeline, gas, pipeline, shut |
| 2 | "Lunch" + "pizza on friday" | lunch, pizza, friday |

**Vocabulary** (every unique word gets a column number):

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| california | friday | gas | lunch | pipeline | pizza | prices | rising | shut |

**Word-count matrix** (rows = emails, columns = words) — this is what BIM and BM25 use:

|  | california | friday | gas | lunch | pipeline | pizza | prices | rising | shut | **length** |
|---|---|---|---|---|---|---|---|---|---|---|
| Email 0 | 1 | 0 | **2** | 0 | 0 | 0 | 2 | 1 | 0 | 6 |
| Email 1 | 0 | 0 | **1** | 0 | **2** | 0 | 0 | 0 | 1 | 4 |
| Email 2 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 3 |

Average length = (6 + 4 + 3) / 3 = **4.33 words**

---

## 6. What a sparse matrix looks like

### The problem
Our real matrix is **251,115 emails × 631,253 words = 158.5 billion cells**.
But an average email uses only **97 different words**, so **99.98% of cells are 0**.

| Real index | Value |
|---|---|
| Total cells | 158,517,996,595 |
| Non-zero cells | 24,551,088 (**0.0155%**) |
| Stored as a normal (dense) matrix | **~1.27 TB** — impossible on a laptop |
| Stored as a sparse matrix | **~300 MB** |

### The solution: store only the non-zero values
Here is the TF-IDF matrix of our worked example as a normal ("dense") table:

```
             california friday  gas  lunch pipeline pizza prices rising shut
             (col 0)   (1)    (2)   (3)    (4)     (5)   (6)    (7)   (8)
Email 0  [   0.35      0     0.53    0      0       0    0.69   0.35   0   ]
Email 1  [   0         0     0.32    0     0.85     0    0      0     0.42 ]
Email 2  [   0        0.58   0      0.58    0      0.58  0      0      0   ]
```

**CSR format** (Compressed Sparse Row — used for TF-IDF) stores it as **3 small arrays**:

```
data    = [0.53, 0.69, 0.35, 0.35 | 0.32, 0.85, 0.42 | 0.58, 0.58, 0.58]   ← the non-zero values
indices = [  2,    6,    7,    0  |   2,    4,    8  |   3,    5,    1 ]   ← their column (word) numbers
indptr  = [0, 4, 7, 10]                                                    ← where each row starts in data
           │  │  │  └ end
           │  │  └ Email 2 = data[7:10]
           │  └ Email 1 = data[4:7]
           └ Email 0 = data[0:4]
```

Reading Email 1: `data[4:7] = [0.32, 0.85, 0.42]` at columns `[2, 4, 8]` → gas = 0.32, pipeline = 0.85, shut = 0.42.
27 cells shrink to 10 values — and for the real data, 158 billion cells shrink to 24.5 million.

**CSC format** (Compressed Sparse Column — used for BIM and BM25) is the same idea but **column by column**.
It answers *"which emails contain this word?"* instantly:

```
word "gas"      → emails [0, 1] with counts [2, 1]
word "pipeline" → emails [1]    with counts [2]
```

This is exactly an **inverted index** (word → list of emails), like the index at the back of a book.
That's why BIM and BM25 only have to look at emails that contain the query words.

---

## 7. Algorithm 1 — TF-IDF + Cosine Similarity

**Question it answers:** *how important is this word to this email?*

- **TF (Term Frequency)** — how many times the word appears in the email.
- **IDF (Inverse Document Frequency)** — how rare the word is across all emails.
  scikit-learn uses the smoothed version: `IDF = ln((1 + N) / (1 + df)) + 1`
  (N = number of emails, df = number of emails containing the word)
- **TF-IDF = TF × IDF** → high when the word is frequent *here* but rare *overall*.
- Each email's vector is then **normalised to length 1** (L2 normalisation).

### Step by step for the query `gas pipeline`

**Step 1 — IDF.** "gas" is in 2 of 3 emails, "pipeline" in only 1:
```
IDF(gas)      = ln(4/3) + 1 = 1.288     (common  → lower)
IDF(pipeline) = ln(4/2) + 1 = 1.693     (rare    → higher)
```

**Step 2 — TF × IDF for Email 1** (gas ×1, pipeline ×2, shut ×1):
```
gas      = 1 × 1.288 = 1.288
pipeline = 2 × 1.693 = 3.386
shut     = 1 × 1.693 = 1.693
length   = √(1.288² + 3.386² + 1.693²) = 3.98
normalised → gas 0.32, pipeline 0.85, shut 0.42      (the row in section 6)
```

**Step 3 — the query becomes a vector too:** `gas 0.61, pipeline 0.80`

**Step 4 — cosine similarity** = multiply matching words and add up (vectors are already length 1):
```
Email 1: 0.32 × 0.61 (gas) + 0.85 × 0.80 (pipeline) = 0.869   ← rank 1
Email 0: 0.53 × 0.61 (gas)                          = 0.319   ← rank 2
Email 2: no matching words                          = 0       ← not shown
```

In code (`search_engine.py`) this whole step is one line — a matrix multiplication against all emails at once:
```python
scores = (matrix @ query_vector.T).toarray().ravel()
```

**Weaknesses:**
- **No saturation:** a word repeated 50 times keeps adding score (keyword stuffing wins).
- **Strong length bias:** the normalisation step divides by the email's length, so *very short*
  emails get a big boost. A 4-word email containing the query word can beat a detailed 300-word email.

---

## 8. Algorithm 2 — BIM (Binary Independence Model)

**Question it answers:** *does this email contain the query words — and how rare are they?*

BIM (Robertson & Spärck Jones, 1976) is the classic **probabilistic** model and the direct ancestor of BM25.

- Each email is treated as a **set of words**: a word is either present (1) or absent (0).
- **How often** a word appears and **how long** the email is are both ignored.
- Score = sum of the weights of the query words the email contains:

```
score = Σ  IDF(word)      over the query words that appear in the email
```

### Where the weight comes from
The original BIM weight needs to know which emails are relevant:
```
w = log[ p (1 − u) / ( u (1 − p) ) ]
  p = chance the word appears in a RELEVANT email       ← unknown: nobody has labelled anything
  u = chance the word appears in a NON-relevant email
```
With no labels, assume **p = 0.5**, and since almost every email is non-relevant, **u ≈ df / N**.
The formula then becomes an IDF:
```
w = log( (N − df + 0.5) / (df + 0.5) )
```
We add **+1 inside the log**, exactly like BM25, so words in more than half the emails don't get negative
weights. That means **BIM and BM25 use the same IDF**, so the comparison between them is fair.

### Step by step for the query `gas pipeline`
**Step 1 — IDF** (N = 3), same as BM25:
```
IDF(gas)      = ln((3 − 2 + 0.5) / (2 + 0.5) + 1) = 0.470
IDF(pipeline) = ln((3 − 1 + 0.5) / (1 + 0.5) + 1) = 0.981
```
**Step 2 — add the IDF of every query word the email contains:**
```
Email 1: has gas ✔ and pipeline ✔  → 0.470 + 0.981 = 1.451   ← rank 1
Email 0: has gas ✔ (twice — doesn't matter) → 0.470          ← rank 2
Email 2: has neither                → 0                      ← not shown
```
In code (`search_engine.py`) it is the BM25 loop without `tf` and `doc_len`:
```python
rows = counts[:, col].indices          # emails containing the word
scores[rows] += np.log((n_emails - df + 0.5) / (df + 0.5) + 1)
```

**Weaknesses:**
- **Huge ties.** Every email that contains the same query words gets exactly the same score.
  For `stock options`, **1,510 emails tie** for first place. For a one-word query like `bankruptcy`,
  **all 4,718 results tie**, so BIM can't rank them at all.
- **No length handling.** A 46,000-word news digest that mentions each word once scores the same as a
  short email that is actually about the topic, so the tied top results are full of huge digests.
- **Repeats ignored.** An email that says "pipeline" ten times scores the same as one that says it once.

---

## 9. Algorithm 3 — BM25 (Best Matching 25)

BM25 = **BIM + two fixes**: it counts repeats (with saturation) and adjusts for email length.
Compared with TF-IDF, the same two fixes stop keyword stuffing and the bias toward tiny emails.
It is the default ranking in Elasticsearch, Lucene, Solr and SQLite FTS.

```
                    IDF(word) × tf × (k1 + 1)
score = Σ   ──────────────────────────────────────────────
     query   tf + k1 × (1 − b + b × email_len / avg_len)
     words
```

| Part | Meaning | Value we use |
|---|---|---|
| `tf` | times the word appears in the email | — |
| `IDF` | rarity of the word: `ln((N − df + 0.5) / (df + 0.5) + 1)` | — |
| `k1` | **TF saturation** — how fast repeats stop helping (0 = BIM, bigger = closer to TF-IDF) | 0.5 |
| `b` | **length normalisation** — how much long emails are penalised (0 = none, 1 = full) | 0.75 |

### Step by step for the query `gas pipeline`

**Step 1 — IDF** (N = 3):
```
IDF(gas)      = ln((3 − 2 + 0.5) / (2 + 0.5) + 1) = 0.470
IDF(pipeline) = ln((3 − 1 + 0.5) / (1 + 0.5) + 1) = 0.981
```

**Step 2 — look up which emails contain each word** (CSC column = inverted index):
```
"gas"      → Email 0 (tf 2, length 6), Email 1 (tf 1, length 4)
"pipeline" → Email 1 (tf 2, length 4)
```
Email 2 is never touched — that's why BM25 is fast.

**Step 3 — score each (word, email) pair:**
```
gas in Email 0:       0.470 × 2 × 1.5 / (2 + 0.5 × (0.25 + 0.75 × 6/4.33)) = 0.533
gas in Email 1:       0.470 × 1 × 1.5 / (1 + 0.5 × (0.25 + 0.75 × 4/4.33)) = 0.479
pipeline in Email 1:  0.981 × 2 × 1.5 / (2 + 0.5 × (0.25 + 0.75 × 4/4.33)) = 1.191
```

**Step 4 — add up per email:**
```
Email 1: 0.479 + 1.191 = 1.670   ← rank 1
Email 0: 0.533                   ← rank 2
```

### The two fixes

**Fix 1 — TF saturation (k1).** Repeating a word gives diminishing returns:

| Word repeated | TF-IDF weight (linear) | BIM weight (binary) | BM25 weight (saturates, k1 = 0.5) |
|---|---|---|---|
| 1× | 1 | 1 | 1.00 |
| 2× | 2 | 1 | 1.20 |
| 5× | 5 | 1 | 1.36 |
| 10× | 10 | 1 | 1.43 |
| 50× | 50 | 1 | 1.49 (never more than k1 + 1 = 1.5) |

BM25 sits between the other two. With **k1 = 0** the TF part becomes `tf / tf = 1`, which is exactly
**BIM**. With a very large k1 it grows almost linearly, like **TF-IDF**. Our k1 = 0.5 keeps BM25 close to
BIM: a repeat helps a little, but matching another *different* query word helps much more.

**Fix 2 — gentle length normalisation (b).** Each email's length is compared to the *average*
length (`email_len / avg_len`), and with `b = 0.75` long emails are only partly penalised —
so a detailed, relevant email is not crushed by a 4-word reply.

**Bonus:** BM25 only touches emails that contain the query words, so in our project it is **~8× faster**
(≈0.005 s vs ≈0.04 s for TF-IDF on 251k emails).

### When does it matter? (`python compare.py`)
In the tiny example all three agree. `compare.py` uses 7 hand-made emails, designed so that each
algorithm's weakness shows, with the query `gas pipeline explosion`:

| Rank | TF-IDF | BIM | BM25 |
|---|---|---|---|
| 1 | **"Explosion"** (one word) | **"Weekly operations report"** (long newsletter) | **"Pipeline explosion near Houston"** ✅ |
| 2 | "Pipeline explosion near Houston" | "Pipeline explosion near Houston" | "Weekly operations report" |
| Why #1 | a tiny email gets a huge cosine | the only email with all 3 words, each mentioned once in passing | the words repeat in a normal-length email that is really about the topic |

---

## 10. What is pickle and how does it work?

**Pickle** is Python's built-in way to **save Python objects to a file and load them back exactly as they were**.
This is called **serialisation** (object → bytes) and **deserialisation** (bytes → object).

### Why we need it
Building the index takes ~75 seconds (read 1.4 GB, parse, build matrices).
We don't want to do that every time the app starts — so we build once, save, and reload in ~1.3 seconds.

### How it's used in this project
```python
# load_data.py — SAVE (once)
with open("index.pkl", "wb") as f:                        # "wb" = write binary
    pickle.dump((emails, tfidf_index, bm25_index), f)

# app.py — LOAD (every time the app starts)
with open("index.pkl", "rb") as f:                        # "rb" = read binary
    emails, tfidf_index, bm25_index = pickle.load(f)
```

### What's inside `index.pkl` (1.1 GB)
```
(
  emails,                      → list of 251,115 dicts {from, to, date, subject, body}
  (vectorizer, tfidf_matrix),  → vocabulary + IDF values, and the CSR sparse matrix
  (counter, counts, doc_len),  → vocabulary, the CSC count matrix, and email lengths (used by BIM and BM25)
)
```

### What pickle actually writes
Pickle turns an object into a compact list of instructions in bytes. For example:
```python
pickle.dumps({"subject": "Gas"})
# b'\x80\x05\x95\x14...}\x94\x8c\x07subject\x94\x8c\x03Gas\x94s.'
#   └ protocol 5   └ "make a dict"  └ 7-char string "subject"  └ 3-char string "Gas"  └ "set item, stop"
```
When loading, Python follows those instructions to rebuild the exact same objects — including
NumPy arrays, SciPy sparse matrices and scikit-learn vectorizers.

### Why not JSON or a database?
| Option | Problem |
|---|---|
| JSON | Can't store NumPy arrays, sparse matrices or scikit-learn objects |
| SQL database | Matrices don't fit in tables naturally; we'd rebuild them on every start |
| **pickle** | ✅ One line to save, one line to load, keeps everything exactly as is |

**Safety note:** loading a pickle can run code, so never load a `.pkl` file from an untrusted source.
Here we create the file ourselves, so it's safe.

---

## 11. Cold start — how the project handles it

"Cold start" means three different things here.

### 1. No training data (no labels, no click history)
In recommender systems, cold start means *there is no history to learn from yet*. All three of our
algorithms are **unsupervised**: they rank using only statistics of the emails themselves
(word counts, document frequency, lengths). So they work on the **very first query**.

| Method | Needs | Works on query #1? |
|---|---|---|
| TF-IDF | word counts + document frequency | ✅ |
| BIM | in theory **relevance labels**; in practice nothing (see below) | ✅ |
| BM25 | word counts + document frequency + email lengths | ✅ |

**BIM is the textbook example.** Its original weight needs `p`, the chance a word appears in a
*relevant* email, which nobody knows at the start. Assuming **p = 0.5** and **u ≈ df/N** turns the weight
into **IDF** (section 8). So **IDF is BIM's answer to cold start**. If users later marked results as
relevant, `p` could be estimated from those marks. This is called **relevance feedback**
(Manning et al., *Introduction to IR*, ch. 11.3.4).

Machine-learned rankers (learning-to-rank, fine-tuned BERT) *do* have a cold-start problem: they need
labelled queries or click logs before they work well.

### 2. App startup (the system's own cold start)
Building the index from `emails.csv` takes ~75 s. We do it **once** in `load_data.py` and pickle the
result; `app.py` loads `index.pkl` in **~1.5 s** at startup. After that every query takes 0.001–0.05 s.

### 3. New emails and unknown words (a limitation)
- A **new email** only appears after `load_data.py` is run again, because the vocabulary, IDF and average
  length are fixed when the index is built.
- A **query word not in the vocabulary** is skipped (`if col is None: continue`). If no query word is
  known, the result list is empty.
- Real systems (Elasticsearch/Lucene) add new documents as small index **segments** that are merged in
  the background, so no full rebuild is needed.

---

## 12. Results and insights (what the searches show)

Real results on all 251,115 emails:

| Query | TF-IDF #1 result | BIM #1 result | BM25 #1 result |
|---|---|---|---|
| `stock options` | "Options" (9 words) | 1,510-way tie (e.g. a 3,004-word market digest) | **"ENE Options"** / "2001 Special Stock Option Grant Awards" |
| `power outage` | "Outage" (4 words) | 643-way tie | **"Planned Power Outage at 3WTC on Sunday"** |
| `natural gas price` | "RE:" (4 words) | 3,305-way tie ("Enron Mentions", 2,684 words) | **"Where will Natural Gas prices go next?"** |
| `sec investigation` | empty subject (7 words) | 317-way tie ("Enron Mentions", 29,543 words) | **"SEC investigation of stock buying"** |
| `skilling resign` | "Skilling's office" (1 word!) | 50-way tie ("Enron Mentions", 8,643 words) | **"FW: Skilling in the news"** / "FW: WSJ Skilling Interview" |
| `bankruptcy` | empty subject (4 words) | **all 4,718 results tie** — no ranking at all | **"Bankruptcy in California; Implications…"** |
| `arthur andersen audit` | "Arthur Andersen audit starts" | 167-way tie ("Enron Mentions (Part I)", 46,669 words) | "Arthur Andersen audit starts" (same as TF-IDF) |

### Insights

1. **TF-IDF is biased toward tiny emails.** Its top results are repeatedly 1–10 word emails. The
   normalisation divides by the email's length, so a short email that happens to contain the query word
   looks like a "perfect match" even when it says almost nothing.
2. **BM25 returns informative emails.** Its top results are typically 100–600 words long and are
   *about* the topic — because length is compared to the average, not punished fully.
3. **BIM can't separate emails that match the same words.** Hundreds or thousands of emails tie for
   first place, so its "top 10" is effectively random — and often huge daily news digests
   ("Enron Mentions"), which contain almost every word. For one-word queries it can't rank at all.
   This shows exactly what BM25 adds on top of BIM: **counting repeats** and **length normalisation**.
4. **BM25 rewards matching *more* query words.** With saturation, matching both "power" *and* "outage"
   beats repeating "outage" many times. TF-IDF returned *SAP system* outages for "power outage"; BM25 returned *power* outages.
5. **When the query is specific, TF-IDF and BM25 agree** (`arthur andersen audit`). BM25 is not magic —
   it is more *reliable*, especially for short, general queries.
6. **All three are keyword-based — none understands meaning.** `shredding documents` returns "Lunch"
   emails at the top, because the common word "documents" dominates. Searching
   "layoffs" will not find emails that only say "job cuts". This needs **semantic search** (future scope).
7. **Same number of results, different order.** All three find exactly the same set of emails (any email
   containing a query word) — the difference is entirely in **ranking**, which is what users actually see.
8. **Data quality matters.** ~50% of the raw dataset was duplicates; without removing them the result
   list showed the same email 3–5 times.

---

## 13. Other search algorithms (and why we didn't use them)

### Keyword (lexical) based
| Algorithm | Idea | Why not used |
|---|---|---|
| **Boolean search** | Exact match with AND / OR / NOT, no ranking | Returns thousands of unranked results — no "best first" |
| **Plain TF (word count)** | Score = how often the word appears | Common words dominate; no rarity (IDF) |
| **TF-IDF + cosine** | ✅ Used — baseline | — |
| **BIM (Binary Independence Model)** | ✅ Used — classic probabilistic model, BM25's ancestor | — |
| **BM25** | ✅ Used — industry standard | — |
| **BM25F** | BM25 with different weights per field (subject > body) | Extra tuning; good future improvement |
| **BM25+** | Fixes BM25 slightly under-scoring very long documents | Marginal gain for our data |
| **Query Likelihood (Dirichlet LM)** | Probability that the email's "language model" generates the query | Similar quality to BM25, harder to explain |
| **DFR (Divergence From Randomness)** | Scores how far a word's frequency is from random chance | Similar quality, more complex math |
| **Fuzzy search / Levenshtein** | Matches typos ("bankrupcy" → "bankruptcy") | Useful add-on, not a ranking method |
| **PageRank** | Ranks web pages by links between them | Emails have no links |

### Semantic (meaning) based
| Algorithm | Idea | Why not used |
|---|---|---|
| **LSA / LSI** (SVD on TF-IDF) | Finds hidden "topics" so related words match | Older approach, expensive on a 251k × 631k matrix |
| **Word2Vec / GloVe** | Each word becomes a meaning vector | Averaging word vectors loses sentence meaning |
| **Sentence embeddings (BERT, Sentence-Transformers)** + vector DB (FAISS, Pinecone, Chroma) | Understands meaning: "layoffs" finds "job cuts" | Needs a model download and GPU/time to embed 251k emails; harder to explain |
| **Hybrid search** (BM25 + embeddings) | Combine keyword and meaning scores | Best of both — the natural next step |
| **Cross-encoder re-ranking** | AI model re-ranks BM25's top 100 | Slow per query, needs a model |

---

## 14. Likely professor questions — and answers

**Q: Why did you choose TF-IDF, BIM and BM25?**
They are the three classic families: TF-IDF is the **vector space** baseline, BIM is the original
**probabilistic** model, and BM25 is BIM extended with term frequency and length — what real search
engines (Elasticsearch, Lucene, Solr) use. Implementing all three lets us *measure* each improvement
on real data instead of just claiming it.

**Q: Why did you write BIM and BM25 yourself but use scikit-learn for TF-IDF?**
scikit-learn has a standard TF-IDF but no BIM or BM25. Writing them by hand (~15 lines each) also shows we
understand every part of the formulas — IDF, `k1`, `b` and length normalisation.

**Q: What is BIM and why is it worse than BM25?**
BIM only checks whether each query word is **present or absent** and adds up the IDFs. It ignores how
often the word appears and how long the email is, so hundreds of emails tie (1,510 for `stock options`)
and long digests fill the top. BM25 = BIM + term frequency (with saturation) + length normalisation.

**Q: Why is BM25 better than TF-IDF?**
Two fixes: (1) **term-frequency saturation** — repeating a word has diminishing returns;
(2) **length normalisation against the average length** — long, informative emails are not crushed.
Our results show TF-IDF's top hits are 1–10 word emails, while BM25's are real, on-topic emails.

**Q: What are k1 and b? Why 0.5 and 0.75?**
`k1` controls saturation: 0 = only presence/absence counts (that is BIM), higher = more linear (closer to TF-IDF).
We use **0.5**, lower than Lucene's default of 1.2, so a word's weight can grow at most 1.5× however often
it repeats. An email that matches *more different* query words wins over one that repeats a single word.
`b` controls length normalisation (0 = none, 1 = full); 0.75 is the standard default.

**Q: How does your system handle cold start?**
All three rankers are unsupervised, so they work from the first query with no labels or click history.
BIM normally needs relevance information; with none, we assume p = 0.5 and it becomes IDF (section 11).
For startup speed we build the index once and pickle it, so the app starts in ~1.5 s instead of ~75 s.
Limitation: new emails need an index rebuild.

**Q: Why cosine similarity and not Euclidean distance?**
Euclidean distance is dominated by vector length (long emails are "far" from short queries). Cosine
measures only the **direction** — *what* the email is about, not *how long* it is.

**Q: Why remove stop words?**
Words like "the", "is", "and" appear in almost every email, carry no meaning, and make the index bigger.

**Q: What is a sparse matrix and why use it?**
A matrix that stores only non-zero values (section 6). Our matrix has 158 billion cells but only
24.5 million non-zero ones — dense it would need ~1.27 TB, sparse it needs ~300 MB.

**Q: What is the difference between CSR and CSC?**
Both store only non-zero values. CSR groups them **by row (email)** — fast for "give me this email's
vector", used for TF-IDF's matrix multiplication. CSC groups them **by column (word)** — fast for
"which emails contain this word?", used for BIM and BM25 (this is an inverted index).

**Q: Why no vector database?**
Our vectors are **sparse keyword vectors** (631k dimensions, mostly zeros) — they fit in RAM as a SciPy
sparse matrix and search in milliseconds. Vector DBs are built for **dense AI embeddings** at millions
of documents — that's our future scope (semantic search).

**Q: Why not just use Elasticsearch?**
Elasticsearch uses BM25 internally — it would be a black box. Building it ourselves shows *how* search
ranking works. For production with millions of emails, Elasticsearch would be the right choice.

**Q: Why Flask and not Django / React?**
We only need two pages. Flask does it in ~50 lines; Django and React add structure and build tools we don't need.

**Q: What is pickle? Is it safe?**
Python's built-in serialisation — it saves Python objects (lists, NumPy arrays, sparse matrices,
scikit-learn objects) to a file and loads them back exactly (section 10). It is only unsafe when
loading files from untrusted sources — here we create the file ourselves.

**Q: How fast is it? Would it scale?**
~0.04 s (TF-IDF) and ~0.001–0.01 s (BIM and BM25) per query on 251k emails, using ~2.3 GB RAM to build.
For millions of emails we'd move to an **inverted index on disk** (Elasticsearch/Lucene).

**Q: What is an inverted index?**
A map from each word → list of documents containing it (like the index at the back of a book).
Our BIM and BM25 use exactly this idea: the CSC sparse matrix gives, for each word, the emails containing it — so we only score those emails.

**Q: How did you evaluate which is better?**
Qualitatively — by comparing top results for real queries and checking which ones are relevant.
A formal evaluation would use **labelled relevance judgements** and metrics like **Precision@10, MAP or NDCG** — future scope.

**Q: What are the limitations?**
- Keyword only — no synonyms or meaning ("layoffs" ≠ "job cuts")
- No typo tolerance
- Whole index must fit in RAM
- No phrase search ("power outage" as an exact phrase)
- No filters (by sender or date)

**Q: Future scope?**
1. Semantic search with sentence embeddings + FAISS
2. Hybrid search (BM25 + embeddings)
3. BM25F — give the subject line more weight than the body
4. Filters by sender / date, phrase search, typo tolerance
5. Formal evaluation with Precision@10 / NDCG

---

## 15. How to run

```bash
cd enron-search
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python test_search.py          # tests → "All tests passed"
python compare.py              # each algorithm puts a different email first
python load_data.py            # build index.pkl (~75 s, only once)
python app.py                  # open http://localhost:8000
```
