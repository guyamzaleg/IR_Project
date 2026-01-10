"""
Test individual ranking methods to verify all 5 requirements are met.
"""
from query_engine import SearchEngine, CONFIG
from Backend.tokenizer import tokenize
import time

def test_all_ranking_methods():
    """Test all 5 required ranking methods."""
    print("\n" + "="*80)
    print("TESTING ALL 5 RANKING METHODS")
    print("="*80 + "\n")
    
    engine = SearchEngine(CONFIG)
    test_query = "python programming language"
    
    print(f"Test query: '{test_query}'\n")
    
    # 1. Cosine similarity using TF-IDF on body
    print("1️⃣  Method: Cosine Similarity (TF-IDF) on Body Text")
    start = time.time()
    results1 = engine.search_body(test_query)
    elapsed1 = time.time() - start
    print(f"   ✓ Returned {len(results1)} results in {elapsed1:.3f}s")
    print(f"   Top 3: {results1[:3]}")
    assert elapsed1 < 35, f"❌ TOO SLOW: {elapsed1:.1f}s > 35s"
    print(f"   {'✅ PASS' if elapsed1 < 35 else '❌ FAIL'}: Time constraint (<35s)\n")
    
    # 2. Binary ranking using title
    print("2️⃣  Method: Binary Ranking on Title")
    start = time.time()
    results2 = engine.search_title(test_query)
    elapsed2 = time.time() - start
    print(f"   ✓ Returned {len(results2)} results in {elapsed2:.3f}s")
    print(f"   Top 3: {results2[:3]}")
    assert elapsed2 < 35, f"❌ TOO SLOW: {elapsed2:.1f}s > 35s"
    print(f"   {'✅ PASS' if elapsed2 < 35 else '❌ FAIL'}: Time constraint (<35s)\n")
    
    # 3. Binary ranking using anchor text
    print("3️⃣  Method: Binary Ranking on Anchor Text")
    start = time.time()
    results3 = engine.search_anchor(test_query)
    elapsed3 = time.time() - start
    print(f"   ✓ Returned {len(results3)} results in {elapsed3:.3f}s")
    print(f"   Top 3: {results3[:3]}")
    assert elapsed3 < 35, f"❌ TOO SLOW: {elapsed3:.1f}s > 35s"
    print(f"   {'✅ PASS' if elapsed3 < 35 else '❌ FAIL'}: Time constraint (<35s)\n")
    
    # 4. Ranking by PageRank
    print("4️⃣  Method: PageRank Ranking")
    start = time.time()
    test_doc_ids = [12, 39, 290, 315, 866]
    pr_scores = engine.get_pagerank(test_doc_ids)
    elapsed4 = time.time() - start
    print(f"   ✓ Returned {len(pr_scores)} PageRank scores in {elapsed4:.3f}s")
    print(f"   Doc IDs: {test_doc_ids}")
    print(f"   PR Scores: {[f'{s:.6f}' for s in pr_scores]}")
    assert elapsed4 < 35, f"❌ TOO SLOW: {elapsed4:.1f}s > 35s"
    print(f"   {'✅ PASS' if elapsed4 < 35 else '❌ FAIL'}: Time constraint (<35s)\n")
    
    # 5. Ranking by PageViews
    print("5️⃣  Method: PageView Ranking")
    start = time.time()
    pv_counts = engine.get_pageviews(test_doc_ids)
    elapsed5 = time.time() - start
    print(f"   ✓ Returned {len(pv_counts)} PageView counts in {elapsed5:.3f}s")
    print(f"   Doc IDs: {test_doc_ids}")
    print(f"   PV Counts: {pv_counts}")
    assert elapsed5 < 35, f"❌ TOO SLOW: {elapsed5:.1f}s > 35s"
    print(f"   {'✅ PASS' if elapsed5 < 35 else '❌ FAIL'}: Time constraint (<35s)\n")
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    all_passed = all(t < 35 for t in [elapsed1, elapsed2, elapsed3, elapsed4, elapsed5])
    
    print(f"\nAll 5 ranking methods implemented: ✅")
    print(f"All methods under 35s constraint: {'✅ YES' if all_passed else '❌ NO'}")
    
    print(f"\nTiming breakdown:")
    print(f"  Body (TF-IDF):    {elapsed1:.3f}s")
    print(f"  Title (Binary):   {elapsed2:.3f}s")
    print(f"  Anchor (Binary):  {elapsed3:.3f}s")
    print(f"  PageRank:         {elapsed4:.3f}s")
    print(f"  PageViews:        {elapsed5:.3f}s")
    
    print("\n" + "="*80)
    print(f"{'✅ ALL TESTS PASSED!' if all_passed else '⚠️  SOME TESTS FAILED'}")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    test_all_ranking_methods()
