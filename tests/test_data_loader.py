# test_data_loader.py
from Backend.data_Loader import load_index, load_pagerank

print("Testing data_loader.py...")
print("-" * 40)

# Test index loading
idx = load_index()
print(f"Index has {len(idx.df)} terms")
print(f"Index has {len(idx.posting_locs)} posting locations")

# Test PageRank loading
pr = load_pagerank()
print(f"PageRank has {len(pr)} documents")
if pr:
    sample_doc_id = list(pr.keys())[0]
    print(f"Sample: doc_id={sample_doc_id}, pagerank={pr[sample_doc_id]:.6f}")

print("-" * 40)
print("✅ data_loader.py works!")
