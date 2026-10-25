# Enron Email Search — TF-IDF vs BM25

## 1. What is this project?

A search engine (like Gmail's search bar) over the **Enron email dataset**. You type a query such as
`stock options`, and it returns the most relevant emails — ranked by **two algorithms side by side**:
**TF-IDF** and **BM25** — so we can compare how well each one ranks.

**Goal:** understand how text search engines work internally, implement two classic ranking
algorithms, and show with real data why **BM25** (used by Elasticsearch, Lucene, Solr) is better than plain TF-IDF.

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
│ 5a. TF-IDF  │ │ 5b. BM25         │
│ Tfidf-      │ │ CountVectorizer  │
│ Vectorizer  │ │ → word counts    │
│ → TF-IDF    │ │ + email lengths  │
│ vectors     │ │                  │
│ (CSR sparse │ │ (CSC sparse      │
│  matrix)    │ │  matrix)         │
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
        │                                  └──────┬───────────────┬───────┘
        │                                         ▼               ▼
        │                          ┌────────────────────┐ ┌────────────────────┐
        │                          │ search_tfidf()     │ │ search_bm25()      │
        │                          │ query → vector     │ │ for each query word│
        │                          │ cosine similarity  │ │  find emails with  │
        │                          │ with ALL emails    │ │  it, add BM25 score│
        │                          └─────────┬──────────┘ └─────────┬──────────┘
        │                                    ▼                      ▼
        │                          ┌──────────────────────────────────────────┐
        │                          │ rank(): keep scores > 0, sort high → low │
        │                          │ → [(email_id, score), ...]  take top 10  │
        │                          └────────────────────┬─────────────────────┘
        │                                               ▼
        │                          ┌──────────────────────────────────────────┐
        │                          │ templates/index.html (Jinja2)            │
        │   HTML page              │ two columns: TF-IDF | BM25               │
        └───────────────────────── │ highlight() wraps query words in <mark>  │
                                   └──────────────────────────────────────────┘

Click a result → GET /email/<id> → show_email() → full email page with highlights
```

---

## 4. Tech stack

| Layer | Technology | Why |
|---|---|---|
| Language | **Python 3** | Simple, best ecosystem for text/ML |
| Parsing | Python built-in `email`, `csv` | No extra install needed |
| TF-IDF | **scikit-learn** `TfidfVectorizer` | Standard, well-tested implementation |
| BM25 | **Written by hand** with NumPy + scikit-learn `CountVectorizer` | To show we understand the formula |
| Math | **NumPy**, **SciPy** sparse matrices | Fast vector math, memory efficient |
| Storage | `pickle` → `index.pkl` | Build index once, load in ~1 second |
| Web backend | **Flask** | Tiny web framework, ~50 lines |
| Frontend | HTML + CSS (Jinja2 templates) | No build step, easy to read |
| Testing | `test_search.py` (plain asserts) | Checks both algorithms rank correctly |

### Project files
| File | What it does |
|---|---|
| `search_engine.py` | Core logic: parse email, build TF-IDF + BM25 indexes, search |
| `load_data.py` | Reads `emails.csv`, removes duplicates, builds and saves the index |
| `app.py` | Flask web app (search page + single email page + word highlighting) |
| `templates/index.html` | The web page |
| `compare.py` | Small hand-made example showing why BM25 beats TF-IDF |
| `test_search.py` | Tests |

---

## 5. Worked example (used in sections 6–8)

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

**Word-count matrix** (rows = emails, columns = words) — this is what BM25 uses:

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

**CSC format** (Compressed Sparse Column — used for BM25) is the same idea but **column by column**.
It answers *"which emails contain this word?"* instantly:

```
word "gas"      → emails [0, 1] with counts [2, 1]
word "pipeline" → emails [1]    with counts [2]
```

This is exactly an **inverted index** (word → list of emails), like the index at the back of a book.
That's why BM25 only has to look at emails that contain the query words.

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

## 8. Algorithm 2 — BM25 (Best Matching 25)

BM25 = TF-IDF **with two fixes**. It is the default ranking in Elasticsearch, Lucene, Solr and SQLite FTS.

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
| `k1` | **TF saturation** — how fast repeats stop helping | 1.5 |
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
gas in Email 0:       0.470 × 2 × 2.5 / (2 + 1.5 × (0.25 + 0.75 × 6/4.33)) = 0.598
gas in Email 1:       0.470 × 1 × 2.5 / (1 + 1.5 × (0.25 + 0.75 × 4/4.33)) = 0.487
pipeline in Email 1:  0.981 × 2 × 2.5 / (2 + 1.5 × (0.25 + 0.75 × 4/4.33)) = 1.437
```

**Step 4 — add up per email:**
```
Email 1: 0.487 + 1.437 = 1.924   ← rank 1
Email 0: 0.598                   ← rank 2
```

### The two fixes

**Fix 1 — TF saturation (k1).** Repeating a word gives diminishing returns:

| Word repeated | TF-IDF weight (linear) | BM25 weight (saturates) |
|---|---|---|
| 1× | 1 | 1.00 |
| 2× | 2 | 1.43 |
| 5× | 5 | 1.92 |
| 10× | 10 | 2.17 |
| 50× | 50 | 2.43 |

**Fix 2 — gentle length normalisation (b).** Each email's length is compared to the *average*
length (`email_len / avg_len`), and with `b = 0.75` long emails are only partly penalised —
so a detailed, relevant email is not crushed by a 4-word reply.

**Bonus:** BM25 only touches emails that contain the query words, so in our project it is **~8× faster**
(≈0.005 s vs ≈0.04 s for TF-IDF on 251k emails).

### When does it matter? (`python compare.py`)
In the tiny example both agree. With a junk email `"gas gas gas gas gas gas gas gas gas"` and a real,
longer email about a pipeline shutdown, for the query `gas pipeline`:
```
TF-IDF:  1. Re: gas  (junk)        BM25:  1. Pipeline shutdown  (correct)
         2. Pipeline shutdown             2. Re: gas
```

---

## 9. What is pickle and how does it work?

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
  (counter, counts, doc_len),  → vocabulary, the CSC count matrix, and email lengths
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

## 10. Results and insights (what the searches show)

Real results on all 251,115 emails:

| Query | TF-IDF #1 result | BM25 #1 result |
|---|---|---|
| `stock options` | "Options" (9 words) | **"ENE Options"** / "2001 Special Stock Option Grant Awards" |
| `power outage` | "Outage" (4 words) | **"Planned Power Outage at 3WTC on Sunday"** |
| `natural gas price` | "RE:" (4 words) | **"Where will Natural Gas prices go next?"** |
| `sec investigation` | empty subject (7 words) | **"SEC investigation of stock buying"** |
| `skilling resign` | "Skilling's office" (1 word!) | **"FW: WSJ Skilling Interview"** |
| `arthur andersen audit` | "Arthur Andersen audit starts" | "Arthur Andersen audit starts" (same) |

### Insights

1. **TF-IDF is biased toward tiny emails.** Its top results are repeatedly 1–10 word emails. The
   normalisation divides by the email's length, so a short email that happens to contain the query word
   looks like a "perfect match" even when it says almost nothing.
2. **BM25 returns informative emails.** Its top results are typically 100–600 words long and are
   *about* the topic — because length is compared to the average, not punished fully.
3. **BM25 rewards matching *more* query words.** With saturation, matching both "power" *and* "outage"
   beats repeating "outage" many times. TF-IDF returned *SAP system* outages for "power outage"; BM25 returned *power* outages.
4. **When the query is specific, both agree** (`arthur andersen audit`). BM25 is not magic —
   it is more *reliable*, especially for short, general queries.
5. **Both are keyword-based — neither understands meaning.** `shredding documents` returns "Lunch"
   emails at the top in both columns, because the common word "documents" dominates. Searching
   "layoffs" will not find emails that only say "job cuts". This needs **semantic search** (future scope).
6. **Same number of results, different order.** Both find exactly the same set of emails (any email
   containing a query word) — the difference is entirely in **ranking**, which is what users actually see.
7. **Data quality matters.** ~50% of the raw dataset was duplicates; without removing them the result
   list showed the same email 3–5 times.

---

## 11. Other search algorithms (and why we didn't use them)

### Keyword (lexical) based
| Algorithm | Idea | Why not used |
|---|---|---|
| **Boolean search** | Exact match with AND / OR / NOT, no ranking | Returns thousands of unranked results — no "best first" |
| **Plain TF (word count)** | Score = how often the word appears | Common words dominate; no rarity (IDF) |
| **TF-IDF + cosine** | ✅ Used — baseline | — |
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

## 12. Likely professor questions — and answers

**Q: Why did you choose TF-IDF and BM25?**
TF-IDF is the classic textbook baseline; BM25 is what real search engines (Elasticsearch, Lucene, Solr)
use. Implementing both lets us *measure* the improvement on real data instead of just claiming it.

**Q: Why did you write BM25 yourself but use scikit-learn for TF-IDF?**
scikit-learn has a standard TF-IDF but no BM25. Writing BM25 by hand (~15 lines) also shows we
understand every part of the formula — `k1`, `b`, IDF and length normalisation.

**Q: Why is BM25 better?**
Two fixes over TF-IDF: (1) **term-frequency saturation** — repeating a word has diminishing returns;
(2) **length normalisation against the average length** — long, informative emails are not crushed.
Our results show TF-IDF's top hits are 1–10 word emails, while BM25's are real, on-topic emails.

**Q: What are k1 and b? Why 1.5 and 0.75?**
`k1` controls saturation (0 = only presence/absence counts, higher = more linear). `b` controls length
normalisation (0 = none, 1 = full). 1.2–2.0 and 0.75 are the standard defaults from research and Lucene; they work well without tuning.

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
"which emails contain this word?", used for BM25 (this is an inverted index).

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
scikit-learn objects) to a file and loads them back exactly (section 9). It is only unsafe when
loading files from untrusted sources — here we create the file ourselves.

**Q: How fast is it? Would it scale?**
~0.04 s (TF-IDF) and ~0.005 s (BM25) per query on 251k emails, using ~2.3 GB RAM to build.
For millions of emails we'd move to an **inverted index on disk** (Elasticsearch/Lucene).

**Q: What is an inverted index?**
A map from each word → list of documents containing it (like the index at the back of a book).
Our BM25 uses exactly this idea: the CSC sparse matrix gives, for each word, the emails containing it — so we only score those emails.

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

## 13. How to run

```bash
cd enron-search
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python test_search.py          # tests → "All tests passed"
python compare.py              # small example: why BM25 beats TF-IDF
python load_data.py            # build index.pkl (~75 s, only once)
python app.py                  # open http://localhost:8000
```
