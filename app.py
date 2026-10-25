"""Web app: search the Enron emails with TF-IDF, BIM and BM25 side by side. Run load_data.py first."""
import pickle
import re
import time

from flask import Flask, abort, render_template, request
from markupsafe import Markup, escape

from search_engine import search_bim, search_bm25, search_tfidf

app = Flask(__name__)

with open("index.pkl", "rb") as f:
    emails, tfidf_index, bm25_index = pickle.load(f)


def highlight(text, query):
    """Wrap every query word found in text with <mark> (HTML-safe)."""
    text = str(escape(text))
    words = [re.escape(str(escape(w))) for w in query.split()]
    if words:
        pattern = r"\b(" + "|".join(words) + r")\b"
        text = re.sub(pattern, r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return Markup(text)


app.jinja_env.filters["highlight"] = highlight


def timed_search(search, query, index):
    start = time.time()
    results = search(query, index)
    return {"results": results[:10], "found": len(results), "took": time.time() - start}


@app.route("/")
def home():
    query = request.args.get("q", "").strip()
    columns = {}
    if query:
        columns = {
            "TF-IDF": timed_search(search_tfidf, query, tfidf_index),
            "BIM": timed_search(search_bim, query, bm25_index),
            "BM25": timed_search(search_bm25, query, bm25_index),
        }
    return render_template("index.html", query=query, columns=columns, emails=emails)


@app.route("/email/<int:i>")
def show_email(i):
    if i >= len(emails):
        abort(404)
    return render_template("index.html", query=request.args.get("q", ""), email=emails[i], emails=emails)


if __name__ == "__main__":
    app.run(debug=True, port=8000)  # not 5000: macOS AirPlay Receiver uses it
