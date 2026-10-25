"""Core of the project: parse emails, build TF-IDF and BM25 indexes, search them."""
import email

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def parse_email(raw):
    """Turn one raw email (headers + body text) into a dict."""
    msg = email.message_from_string(raw)
    if msg.is_multipart():  # rare in Enron: keep only the first part
        msg = msg.get_payload(0)
    return {
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "date": msg.get("Date", ""),
        "subject": msg.get("Subject", ""),
        "body": msg.get_payload(),
    }


def email_texts(emails):
    return [e["subject"] + " " + e["body"] for e in emails]


def rank(scores):
    """Return [(email_index, score), ...] for every email with score > 0, best first."""
    matches = scores.nonzero()[0]
    ranked = matches[scores[matches].argsort()[::-1]]  # sort high score -> low
    return [(int(i), float(scores[i])) for i in ranked]


# ---------- TF-IDF + cosine similarity ----------

def build_tfidf_index(emails):
    """Turn every email into a TF-IDF vector (one row per email, one column per word)."""
    vectorizer = TfidfVectorizer(stop_words="english")  # drops "the", "is", "and"...
    matrix = vectorizer.fit_transform(email_texts(emails))
    return vectorizer, matrix


def search_tfidf(query, index):
    vectorizer, matrix = index
    query_vector = vectorizer.transform([query])
    # Rows are already length-normalised, so a dot product = cosine similarity
    scores = (matrix @ query_vector.T).toarray().ravel()
    return rank(scores)


# ---------- BIM (Binary Independence Model) ----------

def search_bim(query, index):
    """
    score(email) = sum of IDF(word) over query words the email contains.

    Binary: a word is either present (1) or absent (0) - repeats and email
    length are ignored. Uses the BM25 index (it already knows which emails
    contain each word). Same IDF as BM25, so BM25 = BIM + tf + length.
    """
    counter, counts, doc_len = index
    n_emails = counts.shape[0]
    scores = np.zeros(n_emails)

    for word in counter.build_analyzer()(query):
        col = counter.vocabulary_.get(word)
        if col is None:
            continue
        rows = counts[:, col].indices  # emails containing the word (tf not used)
        df = len(rows)
        scores[rows] += np.log((n_emails - df + 0.5) / (df + 0.5) + 1)

    return rank(scores)


# ---------- BM25 ----------

def build_bm25_index(emails):
    """Count how often each word appears in each email, and each email's length."""
    counter = CountVectorizer(stop_words="english")
    counts = counter.fit_transform(email_texts(emails)).tocsc()  # csc = fast column (word) lookup
    doc_len = counts.sum(axis=1).A1  # number of words in each email
    return counter, counts, doc_len


def search_bm25(query, index, k1=0.5, b=0.75):
    """
    score(email) = sum over query words of:
        IDF(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))

    k1 -> TF saturation: the 10th repeat of a word adds much less than the 1st
    b  -> length normalisation: long emails are penalised, but only gently
    """
    counter, counts, doc_len = index
    n_emails = counts.shape[0]
    avg_len = doc_len.mean()
    scores = np.zeros(n_emails)

    for word in counter.build_analyzer()(query):  # same tokenising as the index
        col = counter.vocabulary_.get(word)
        if col is None:
            continue  # word never appears in any email
        column = counts[:, col]
        rows, tf = column.indices, column.data  # emails containing the word, and how often
        df = len(rows)  # how many emails contain the word
        idf = np.log((n_emails - df + 0.5) / (df + 0.5) + 1)
        scores[rows] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[rows] / avg_len))

    return rank(scores)
