"""Demo: why BM25 ranks better than TF-IDF. Run: python compare.py"""
from search_engine import build_bm25_index, build_tfidf_index, search_bm25, search_tfidf

EMAILS = [
    {"subject": "Re: gas", "body": "gas gas gas gas gas gas gas gas gas"},  # junk / repetitive
    {"subject": "Pipeline shutdown", "body": (  # the email we actually want
        "Team, the gas pipeline from Houston to Dallas will be shut down this Friday "
        "for scheduled maintenance. Customers should expect delays in deliveries. "
        "The operations desk will send updated schedules on Monday. Please inform "
        "traders and the scheduling group, and contact Mike with any questions "
        "about volumes or contracts.")},
    {"subject": "Gas prices", "body": "Gas prices are rising in California this week."},
    {"subject": "Lunch", "body": "Pizza on Friday?"},
    {"subject": "Meeting notes", "body": "Notes from the trading meeting are attached."},
]
QUERY = "gas pipeline"


def show(title, results):
    print(f"\n{title}")
    for rank, (i, score) in enumerate(results, 1):
        print(f"  {rank}. {score:6.3f}  {EMAILS[i]['subject']}")


if __name__ == "__main__":
    print(f'Query: "{QUERY}"')
    show("TF-IDF (cosine similarity):", search_tfidf(QUERY, build_tfidf_index(EMAILS)))
    show("BM25:", search_bm25(QUERY, build_bm25_index(EMAILS)))

    print("\nWhy? How much a word's score grows when it repeats (same-length email):")
    print("  repeats   TF-IDF (linear)   BM25 (saturates, k1=1.5)")
    for tf in [1, 2, 5, 10, 50]:
        print(f"  {tf:7}   {tf:15}   {tf * 2.5 / (tf + 1.5):.2f}")
