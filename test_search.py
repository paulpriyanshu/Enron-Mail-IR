"""Quick check that parsing and ranking work: python test_search.py"""
from compare import EMAILS, QUERY
from search_engine import build_bm25_index, build_tfidf_index, parse_email, search_bim, search_bm25, search_tfidf

e = parse_email("From: ken@enron.com\nSubject: Gas prices\n\nGas prices in California are rising.")
assert e["from"] == "ken@enron.com" and e["subject"] == "Gas prices"

emails = [
    e,
    {"subject": "Lunch", "body": "Pizza on friday?"},
    {"subject": "Re: gas", "body": "I agree, gas gas gas everywhere."},
]
for build, search in [(build_tfidf_index, search_tfidf), (build_bm25_index, search_bm25), (build_bm25_index, search_bim)]:
    index = build(emails)
    assert sorted(i for i, _ in search("gas", index)) == [0, 2]  # pizza not matched
    assert search("banana", index) == []  # unknown word -> no results

# More "gas" ranks higher in TF-IDF and BM25; BIM ignores repeats, so it's a tie
assert search_bm25("gas", build_bm25_index(emails))[0][0] == 2
bim = search_bim("gas", build_bm25_index(emails))
assert bim[0][1] == bim[1][1]

# The presentation demo: each method puts a different email first
assert search_tfidf(QUERY, build_tfidf_index(EMAILS))[0][0] == 0  # tiny "Explosion"
assert search_bim(QUERY, build_bm25_index(EMAILS))[0][0] == 2  # long newsletter
assert search_bm25(QUERY, build_bm25_index(EMAILS))[0][0] == 1  # real report
print("All tests passed")
