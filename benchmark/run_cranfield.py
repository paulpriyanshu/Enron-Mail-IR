"""Run from the project root:  pip install rank_bm25  &&  python benchmark/run_cranfield.py

Benchmark the project's own TF-IDF and BM25 code (plus other classic rankers)
on the Cranfield test collection (1,400 docs, 225 queries, human relevance labels)."""
import json
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25L, BM25Plus
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # project root, for search_engine.py
from search_engine import build_bm25_index, build_tfidf_index, search_bm25, search_tfidf  # noqa

D = os.path.join(HERE, "data") + os.sep
raw = open(D + "cran.all.1400.xml").read()
docs = []
for m in re.finditer(r"<docno>(.*?)</docno>.*?<text>(.*?)</text>", raw, re.S):
    docs.append({"id": int(m.group(1)), "subject": "", "body": m.group(2)})
qraw = open(D + "cran.qry.xml").read()
queries = [re.sub(r"\s+", " ", t).strip() for t in re.findall(r"<title>(.*?)</title>", qraw, re.S)]
qrels = {}
for line in open(D + "cranqrel.trec.txt"):
    q, _, d, r = line.split()
    if int(r) > 0:
        qrels.setdefault(int(q), set()).add(int(d))
print(len(docs), "docs,", len(queries), "queries,", sum(len(v) for v in qrels.values()), "judgments")
id_of = [d["id"] for d in docs]


def metrics(ranked_ids, rel):
    top10 = ranked_ids[:10]
    gains = [1 if d in rel else 0 for d in top10]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), 10)))
    hits, ap = 0, 0.0
    for i, d in enumerate(ranked_ids[:1000]):
        if d in rel:
            hits += 1
            ap += hits / (i + 1)
    rr = next((1 / (i + 1) for i, d in enumerate(ranked_ids[:10]) if d in rel), 0)
    return {
        "ndcg10": dcg / idcg if idcg else 0,
        "p10": sum(gains) / 10,
        "map": ap / len(rel),
        "mrr10": rr,
        "r100": len(set(ranked_ids[:100]) & rel) / len(rel),
    }


tok = CountVectorizer(stop_words="english").build_analyzer()
doc_toks = [tok(d["body"]) for d in docs]
doc_len = np.array([len(t) for t in doc_toks])


def evaluate(name, search_fn):
    res, lens, t = [], [], 0.0
    for qi, q in enumerate(queries, 1):
        rel = qrels.get(qi)
        if not rel:
            continue
        s = time.perf_counter()
        ranked = search_fn(q)  # list of doc positions, best first
        t += time.perf_counter() - s
        res.append(metrics([id_of[i] for i in ranked], rel))
        lens.extend(doc_len[ranked[:10]])
    out = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    out["ms_per_query"] = 1000 * t / len(res)
    out["avg_len_top10"] = float(np.mean(lens))
    out["n_queries"] = len(res)
    print(f"{name:28s} " + " ".join(f"{k}={v:.3f}" for k, v in out.items()))
    return out


R = {}
# 1. Boolean AND (match all query words, no ranking -> doc order)
inv = {}
for i, t in enumerate(doc_toks):
    for w in set(t):
        inv.setdefault(w, set()).add(i)


def boolean_and(q):
    ws = [w for w in tok(q) if w in inv]
    if not ws:
        return []
    s = set.intersection(*[inv[w] for w in ws])
    return sorted(s)


def boolean_or(q):
    s = set().union(*[inv.get(w, set()) for w in tok(q)])
    return sorted(s)


R["Boolean AND"] = evaluate("Boolean AND", boolean_and)
R["Boolean OR"] = evaluate("Boolean OR", boolean_or)

# 2. Raw term frequency (count dot product, no IDF, no normalisation)
cv = CountVectorizer(stop_words="english")
C = cv.fit_transform([d["body"] for d in docs])


def raw_tf(q):
    s = (C @ cv.transform([q]).T).toarray().ravel()
    nz = s.nonzero()[0]
    return list(nz[s[nz].argsort()[::-1]])


R["Raw TF"] = evaluate("Raw TF (no IDF)", raw_tf)

# 3. Project TF-IDF (exact project code)
tf_idx = build_tfidf_index(docs)
R["TF-IDF (project)"] = evaluate("TF-IDF cosine (project)", lambda q: [i for i, _ in search_tfidf(q, tf_idx)])

# 4. Project BM25 (exact project code, k1=1.5, b=0.75)
bm_idx = build_bm25_index(docs)
R["BM25 (project)"] = evaluate("BM25 (project)", lambda q: [i for i, _ in search_bm25(q, bm_idx)])

# 5. BM25 variants
bl = BM25L(doc_toks)
bp = BM25Plus(doc_toks)


def via(model):
    def f(q):
        s = model.get_scores(tok(q))
        nz = np.nonzero(s)[0]
        return list(nz[s[nz].argsort()[::-1]])
    return f


R["BM25L"] = evaluate("BM25L", via(bl))
R["BM25+"] = evaluate("BM25+", via(bp))

# 6. Query likelihood, Dirichlet smoothing (mu = 1000, Anserini default)
counts = C.tocsc()
coll = np.asarray(C.sum(axis=0)).ravel()
coll_p = coll / coll.sum()
dl = np.asarray(C.sum(axis=1)).ravel()


def qld(q, mu=1000):
    words = [cv.vocabulary_[w] for w in tok(q) if w in cv.vocabulary_]
    if not words:
        return []
    score = np.zeros(C.shape[0])
    touched = np.zeros(C.shape[0], bool)
    for c in words:
        col = counts[:, c]
        tf = np.zeros(C.shape[0])
        tf[col.indices] = col.data
        touched[col.indices] = True
        score += np.log((tf + mu * coll_p[c]) / (dl + mu))
    idx = np.where(touched)[0]
    return list(idx[score[idx].argsort()[::-1]])


R["Query likelihood (Dirichlet)"] = evaluate("Query likelihood (LM-Dir)", qld)

# 7. LSA: TF-IDF -> 200-dim SVD, cosine
X = tf_idx[1]
svd = TruncatedSVD(200, random_state=0)
Z = normalize(svd.fit_transform(X))


def lsa(q):
    zq = normalize(svd.transform(tf_idx[0].transform([q])))
    s = Z @ zq.ravel()
    return list(np.argsort(-s)[:1000])


R["LSA (200 dims)"] = evaluate("LSA / LSI (200 dims)", lsa)

# 8. BM25 parameter sweep
sweep = {}
for k1 in [0.5, 0.9, 1.2, 1.5, 2.0]:
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        res = []
        for qi, q in enumerate(queries, 1):
            rel = qrels.get(qi)
            if rel:
                res.append(metrics([id_of[i] for i, _ in search_bm25(q, bm_idx, k1=k1, b=b)], rel)["ndcg10"])
        sweep[f"{k1}|{b}"] = float(np.mean(res))
print("sweep", json.dumps(sweep))

# 9. Per-query wins
per = {"bm25": 0, "tfidf": 0, "tie": 0}
for qi, q in enumerate(queries, 1):
    rel = qrels.get(qi)
    if not rel:
        continue
    a = metrics([id_of[i] for i, _ in search_tfidf(q, tf_idx)], rel)["ndcg10"]
    b_ = metrics([id_of[i] for i, _ in search_bm25(q, bm_idx)], rel)["ndcg10"]
    per["bm25" if b_ > a + 1e-9 else "tfidf" if a > b_ + 1e-9 else "tie"] += 1
print("per-query", per)
json.dump({"results": R, "sweep": sweep, "per_query": per,
           "avg_doc_len": float(doc_len.mean())}, open(os.path.join(HERE, "results.json"), "w"), indent=1)
