"""Quick check that parsing and ranking work: python test_search.py"""
from compare import EMAILS, QUERY
from search_engine import build_bm25_index, build_tfidf_index, parse_email, search_bm25, search_tfidf

e = parse_email("From: ken@enron.com\nSubject: Gas prices\n\nGas prices in California are rising.")
assert e["from"] == "ken@enron.com" and e["subject"] == "Gas prices"

emails = [
    e,
    {"subject": "Lunch", "body": "Pizza on friday?"},
    {"subject": "Re: gas", "body": "I agree, gas gas gas everywhere."},
]
for build, search in [(build_tfidf_index, search_tfidf), (build_bm25_index, search_bm25)]:
    index = build(emails)
    assert [i for i, _ in search("gas", index)] == [2, 0]  # most "gas" first, pizza not matched
    assert search("banana", index) == []  # unknown word -> no results

# The presentation demo: TF-IDF puts the junk email first, BM25 the real one
assert search_tfidf(QUERY, build_tfidf_index(EMAILS))[0][0] == 0
assert search_bm25(QUERY, build_bm25_index(EMAILS))[0][0] == 1
print("All tests passed")
