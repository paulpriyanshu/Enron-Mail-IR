"""Demo: TF-IDF vs BIM vs BM25 - each puts a different email first. Run: python compare.py"""
from search_engine import build_bm25_index, build_tfidf_index, search_bim, search_bm25, search_tfidf

EMAILS = [
    # 0. Tiny, one word - TF-IDF bait (short email = big cosine)
    {"subject": "Explosion", "body": "Explosion!"},
    # 1. The email we actually want - BM25 should win
    {"subject": "Pipeline explosion near Houston", "body": (
        "An explosion hit the pipeline near Houston this morning. Crews shut the pipeline "
        "and are checking the damage. A second explosion was reported but not confirmed. "
        "Field teams will inspect the pipeline again tomorrow and report back to the desk. "
        "Customers in Dallas should expect delays this week.")},
    # 2. Long newsletter that mentions all 3 words once, in passing - BIM bait
    {"subject": "Weekly operations report", "body": (
        "This week the trading floor moved to the new building. Power desk volumes were up. "
        "IT upgraded the email servers and the phones. HR reminds everyone about benefits "
        "enrollment. Gas desk had a quiet week. Legal reviewed the new contracts. "
        "Facilities fixed the elevator. The pipeline group finished its budget. "
        "Safety training covered fire drills and the explosion response plan. "
        "Accounting closed the quarter. The cafeteria menu changes on Monday. "
        "Parking passes are due Friday. Please update your timesheets. "
        "The holiday party is next month. Thanks to everyone for a great week.")},
    {"subject": "Gas prices", "body": "Gas prices are rising in California."},
    {"subject": "Gas storage", "body": "Gas storage levels are low this winter."},
    {"subject": "Lunch", "body": "Pizza on Friday?"},
    {"subject": "Meeting notes", "body": "Notes from the trading meeting are attached."},
]
QUERY = "gas pipeline explosion"


def show(title, results):
    print(f"\n{title}")
    for rank, (i, score) in enumerate(results, 1):
        print(f"  {rank}. {score:6.3f}  {EMAILS[i]['subject']}")


if __name__ == "__main__":
    print(f'Query: "{QUERY}"')
    show("TF-IDF (cosine) - favours the shortest email:", search_tfidf(QUERY, build_tfidf_index(EMAILS)))
    show("BIM (present/absent) - favours the email with the most query words:", search_bim(QUERY, build_bm25_index(EMAILS)))
    show("BM25 (BIM + repeats + length) - favours the real report:", search_bm25(QUERY, build_bm25_index(EMAILS)))

    print("\nHow much a word's score grows when it repeats (same-length email):")
    print("  repeats   TF-IDF (linear)   BIM (binary)   BM25 (saturates, k1=0.5)")
    for tf in [1, 2, 5, 10, 50]:
        print(f"  {tf:7}   {tf:15}   {1:12}   {tf * 1.5 / (tf + 0.5):.2f}")  # tf*(k1+1)/(tf+k1)
