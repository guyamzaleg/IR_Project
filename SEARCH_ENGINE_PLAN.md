# 🔍 Wikipedia Search Engine - Implementation Guide

## Project Overview

This guide provides a **clear, step-by-step workflow** to build a working Wikipedia search engine using the inverted index and PageRank you created in GCP. We'll work with local files for fast development.

**Timeline:** 3-4 hours to working prototype

---

## 📦 What You Built in GCP

| Resource | GCP Location | Description |
|----------|--------------|-------------|
| Inverted Index | `gs://207698911/postings_gcp/index.pkl` | Term statistics + posting locations |
| Posting Lists | `gs://207698911/postings_gcp/*.bin` | Actual posting data (thousands of files) |
| PageRank | `gs://207698911/pr/*.csv.gz` | PageRank scores from GraphFrames |
| Training Queries | Local: `queries_train.json` | Test queries for evaluation |

---

## 🎯 Implementation Workflow

```
1. Download Data     →  Get files from GCP to local machine
2. Create Helpers    →  data_loader.py + tokenizer.py  
3. Implement /search →  Single main search endpoint only
4. Test Locally      →  Run and verify results
5. Improve           →  Optimize scoring and add features
```

**Note:** We'll implement ONLY the main `/search` endpoint to get started quickly. Other endpoints (`/search_body`, `/search_title`, `/search_anchor`) can be added later if needed.

---

## 📥 Step 1: Download Data from GCP

**Estimated time:** 30-60 minutes (depends on internet speed)
**Disk space needed:** ~15-20 GB

### 1.1 Create Data Directory

### 1.1 Download Index Files

```bash
# Create data directory
mkdir -p data/postings_gcp

# Download index.pkl (contains merged posting_locs from all buckets)
gsutil cp gs://207698911/postings_gcp/index.pkl data/postings_gcp/

# Download ALL posting list files (thousands of .bin files)
# ⚠️ CRITICAL: These contain the actual posting lists data!
# Without them, you cannot search - index.pkl just has pointers to these files
gsutil -m cp "gs://207698911/postings_gcp/*.bin" data/postings_gcp/

# Note: You DON'T need the *_posting_locs.pickle files
# Those were already merged into index.pkl in your GCP notebook

# Download PageRank
gsutil cp "gs://207698911/pr/*.csv.gz" data/
```

**Important**: 
- You'll have **thousands of `.bin` files** - this is normal! They contain the actual posting lists data.
- **YOU MUST HAVE THE .BIN FILES** - `index.pkl` only contains pointers to these files, not the data itself
- The `index.pkl` has all posting locations merged (from the notebook's `super_posting_locs`)
- You can ignore/skip the `*_posting_locs.pickle` files

**How it works:**
```
Query "python" → index.pkl says "read bytes 1000-2000 from 5_003.bin"
              → Opens 5_003.bin and reads posting list: [(12345, 5), (67890, 3), ...]
              → Returns matching documents!
```

### 1.6 Verify Your Downloads

Create a quick test script to verify everything downloaded correctly:

```python
# verify_data.py
import os
import pickle

print("="*60)
print("VERIFYING DOWNLOADED DATA")
print("="*60)

# Check index
index_path = "data/postings_gcp/index.pkl"
if os.path.exists(index_path):
    print(f"✅ Index file exists: {os.path.getsize(index_path) / (1024**2):.2f} MB")
else:
    print("❌ Index file NOT found!")

# Check .bin files
---

## 🔧 Step 2: Create Helper Modules

**Estimated time:** 30 minutes

### 2.1 Create `data_loader.py`

This module loads the index and PageRank data at startup.

```python
# data_loader.py
import pickle
import pandas as pd
import os
from inverted_index_gcp import InvertedIndex

def load_index():
    """Load the inverted index from local disk."""
    print("Loading inverted index...")
    index = InvertedIndex.read_index("data/postings_gcp", "index")
    print(f"✓ Index loaded: {len(index.df)} terms")
    return index

def load_pagerank():
    """Load PageRank scores from CSV files."""
    print("Loading PageRank...")
    pr_files = [f"data/{f}" for f in os.listdir("data") if f.endswith('.csv.gz')]
    
    if not pr_files:
        print("⚠ No PageRank files found")
        return {}
    
    dfs = []
    for file in pr_files:
        df = pd.read_csv(file, header=None, names=['doc_id', 'pagerank'])
        dfs.append(df)
    
    pr_df = pd.concat(dfs)
    pr_dict = dict(zip(pr_df['doc_id'].astype(int), pr_df['pagerank']))
    print(f"✓ PageRank loaded: {len(pr_dict)} documents")
    return pr_dicter, requires internet)
        index = InvertedIndex.read_index("postings_gcp", "index", bucket_name=BUCKET_NAME)
    
    print(f"✓ Index loaded: {len(index.df)} terms")
    print(f"✓ Mode: {'LOCAL' if WORK_LOCALLY else 'REMOTE (GCS)'}")
    return index

def load_pagerank():
    """Load PageRank scores from CSV."""
    print("Loading PageRank...")
    pr_fCreate `tokenizer.py`

Use the **exact same tokenizer** from your GCP notebook to ensure consistency.

```python
# tokenizer.py
import re
from nltk.corpus import stopwords

# Exact same stopwords from GCP notebook
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    """
    Tokenize text and remove stopwords.
    Matches the tokenization from GCP notebook exactly.
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in all_stopwords]
```

**Test it:**
```python
from tokenizer import tokenize
print(tokenize("Python programming language"))
# Should output: ['python', 'programming', 'language'
import re
from nltk.corpus import stopwords

# Same stopwords from your GCP notebook
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text):
    """Tokenize text and remove stopwords (same as GCP notebook)."""
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [t for t in tokens if t not in all_stopwords]
```

---

## 🔍 Step 3: Implement Basic Search (1.5 hours)

We'll start with just `/search_body` to get something working quickly.

### 3.1 Update `search_frontend.py`

Add at the top (after imports):

```python
# search_frontend.py
from flask import Flask, request, jsonify, render_template
import os
import math
from collections import defaultdict
from data_loader import load_index, load_pagerank, load_doc_titles
from tokenizer import tokenize

# Load data once at startup
print("="*50)
print("🚀 Starting search engine...")
print("="*50)
inverted_index = load_index()
pagerank_dict = load_pagerank()
print("✓ All data loaded successfully!")
print("="*50)

# Global constants
N_DOCS = 6000000  # Approximate Wikipedia size
```

### 3.2 Implement `/search` Endpoint

Start with the simplest approach - just count matching documents:

```python
@app.route("/search")
def search():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    if not query_tokens:G
        return jsonify(res)
    
    # Count how many query terms each document has
    doc_scores = defaultdict(int)G
    
    for term in query_tokens:
        if term not in inverted_index.posting_locs:
            continue
        
        # Read posting list for this term
        posting_list = inverted_index.read_a_posting_list("data/postings_gcp", term)
        
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += 1  # Simple: count matching terms
    
    # Sort by score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100]
    
    # Return list of doc IDs (as strings, like queries_train.json)
    res = [str(doc_id) for doc_id, _ in sorted_docs]
    # END SOLUTION
    
    return jsonify(res)Search Engine

**EstimaInstall Required Packages

```powershell
# Make sure virtual environment is activated
.venv\Scripts\Activate.ps1

# Install packages
pip install flask pandas nltk

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

**Verify installations:**
```powershell
python -c "import flask, pandas, nltk; print('✅ All packages installed')"
```

### 4.2 Run the Server

```powershell
python search_frontend.py
```

You should see:
```
============================================================
🚀 STARTING SEARCH ENGINE
============================================================
Loading inverted index...
✓ Index loaded: 524283 terms
Loading PageRank...
✓ PageRank loaded: 6348910 documents
✓ All data loaded successfully!
============================================================
 * Running on http://0.0.0.0:8080
```

### 4.3 Test a Query

**Option 1: Browser**
```
http://localhost:8080/search_body?query=python+programming
```Upgrade to TF-IDF Scoring

Replace binary counting with TF-IDF for better relevance:

```python
@app.route("/search_body")
def search_body():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    if not query_tokens:
        return jsonify(res)
    
    doc_scores = defaultdict(float)
    
    for term in query_tokens:
        if term not in inverted_index.posting_locs:
            continue
        
        # Calculate IDF (inverse document frequency)
        df = inverted_index.df[term]
        idf = math.log10(N_DOCS / df) if df > 0 else 0
        
        posting_list = inverted_index.read_a_posting_list("data/postings_gcp", term)
        Implement `/get_pagerank` Endpoint

**Purpose:** Return PageRank scores for given document IDs.

```python
@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 
    
    Test with:
        import requests
        requests.post('http://localhost:8080/get_pagerank', json=[1,5,8])
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    res = [pagerank_dict.get(int(doc_id), 0.0) for doc_id in wiki_ids]
    # END SOLUTION
    
    return jsonify(res)
```

**Test it:**
```python
# test_pagerank.py
import requests

# Test with some doc IDs
response = requests.post('http://localhost:8080/get_pagerank', 
                         json=['12345', '67890', '11111'])
scores = response.json()
print(f"PageRank scores: {scores}")
assert len(scores) == 3, "Should return 3 scores!"
assert all(isinstance(s, float) for s in scores), "Scores should be floats!"
print("✅ PageRank endpoint works!")
```

### 5.3 Implement Main `/search` - Combine Signals

**Why combine signals?**
- Body text alone: Finds relevant content
- PageRank alone: Finds popular/authoritative pages
- Combined: Best of both!

**Weighting:** 80% text relevance, 20% PageRank authority

Implement the main `/search` endpoint:

```python
@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    if not query_tokens:
        return jsonify(res)
    
    # Calculate body text relevance (TF-IDF)
    doc_scores = defaultdict(float)
    for term in query_tokens:
        if term not in inverted_index.posting_locs:
            continue
        
        df = inverted_index.df[term]
        idf = math.log10(N_DOCS / df) if df > 0 else 0
        posting_list = inverted_index.read_a_posting_list("data/postings_gcp", term)
        
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += tf * idf
    
    # Combine with PageRank (weighted)
    final_scores = {}
    for doc_id, text_score in doc_scores.items():
        pr_score = math.log1p(pagerank_dict.get(doc_id, 0))
        # 80% text relevance, 20% PageRank authority
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:100]
    return [str(doc_id) for doc_id, _ in sorted_docs]
```

### 5.2 Add PageRank Endpoint

```python
@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    res = [pagerank_dict.get(int(doc_id), 0.0) for doc_id in wiki_ids]
    # END SOLUTION
    
    return jsonify(res)
```

### 5.3 Combine Body Search + PageRank

```python
@app.route("/search")
def search():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    
    # BEGIN SOLUTION
    query_tokens = tokenize(query)
    if not query_tokens:
        return jsonify(res)
    
    doc_scores = defaultdict(float)
    
    # Body text relevance
    for term in query_tokens:
        if term not in inverted_index.posting_locs:
            continue
        df = inverted_index.df[term]
        idf = math.log10(N_DOCS / df) if df > 0 else 0
        posting_list = inverted_index.read_a_posting_list("data/postings_gcp", term)
        
        for doc_id, tf in posting_list:
            doc_scores[doc_id] += tf * idf
    
    # Combine with PageRank
    final_scores = {}
    for doc_id, text_score in doc_scores.items():
        pr_score = math.log1p(pagerank_dict.get(doc_id, 0))
        final_scores[doc_id] = 0.8 * text_score + 0.2 * pr_score
    
    sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:100]
    res = [str(doc_id) for doc_id, _ in sorted_docs]
    # END SOLUTION
    
    return jsonify(res)
```

---

## 📁 Suggested File Structure

```
IR_Project/
├── search_frontend.py      # Main Flask app
├── inverted_index_gcp.py   # Index reading utilities
├── retrieval/
│   ├── __init__.py
│   ├── tokenizer.py        # Tokenization functions
│   ├── bm25.py             # BM25 implementation
│   ├── tfidf.py            # TF-IDF implementation
│   ├── ranking.py          # Combined ranking logic
│   └── cache.py            # Caching utilities
├── data/
│   ├── index.pkl           # Inverted index
│   ├── pagerank.csv.gz     # PageRank scores
│   ├── pageviews.pkl       # Page view counts
│   ├── doc_titles.pkl      # Document ID to title mapping
│   └── doc_lengths.pkl     # Pre-computed document lengths
├── evaluation/
│   ├── evaluate.py         # Evaluation metrics
│   └── queries_train.json  # Test queries
└── Frontend/
    ├── templates/
    └── static/
```

---

## 📊 Evaluation Strategy

### Metrics to Track
- **MAP (Mean Average Precision)** - Primary metric
- **Precision@5, Precision@10** - Top results quality
- **Recall@100** - Coverage
- **MRR (Mean Reciprocal Rank)** - First relevant result position
- **NDCG** - Graded relevance

### Evaluation Script
```python
def evaluate(search_func, queries_with_relevance):
    """
    Evaluate search function against ground truth.
    """
    metrics = {'map': [], 'p@5': [], 'p@10': []}
    
    for query, relevant_docs in queries_with_relevance.items():
        results = search_func(query)
        result_ids = [doc_id for doc_id, title in results]
        
        # Calculate metrics
        metrics['map'].append(average_precision(result_ids, relevant_docs))
        metrics['p@5'].append(precision_at_k(result_ids, relevant_docs, 5))
        metrics['p@10'].append(precision_at_k(result_ids, relevant_docs, 10))
    
    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

---

## 🚀 Deployment Checklist

- [ ] Load all data files at startup (not per-request)
- [ ] Test all endpoints locally
- [ ] Verify response times < 5 seconds for `/search`
- [ ] Handle edge cases (empty queries, special characters)
- [ ] Add error handling and logging
- [ ] Test with `queries_train.json`
- [ ] Deploy to GCP/Colab

---

## ⚡ Performance Tips

1. **Load data once at startup** - Not per request
2. **Use efficient data structures** - Dict for O(1) lookups
3. **Cache hot posting lists** - Frequent terms in memory
4. **Limit posting list reads** - Early termination
5. **Pre-compute statistics** - Document lengths, norms, etc.
6. **Use NumPy for vector operations** - Faster than pure Python
7. **Profile your code** - Find bottlenecks with cProfile

---

## 📝 Implementation Order (Recommended)

| Step | Task | Endpoint | Est. Time |
|------|------|----------|-----------|
| 1 | Load index & PageRank | Setup | 1 hour |
| 2 | Implement tokenizer | Helper | 30 min |
| 3 | Implement `/get_pagerank` | `/get_pagerank` | 30 min |
| 4 | Implement `/search_title` | `/search_title` | 1 hour |
| 5 | Implement `/search_body` with TF-IDF | `/search_body` | 2 hours |
| 6 | Implement `/search_anchor` | `/search_anchor` | 1 hour |
| 7 | Implement `/get_pageview` | `/get_pageview` | 30 min |
| 8 | Implement main `/search` with BM25 + hybrid | `/search` | 3 hours |
| 9 | Optimize & cache | All | 2 hours |
| 10 |Your File Structure

```
IR_Project/
├── search_frontend.py           # Flask app (EDIT THIS)
├── inverted_index_gcp.py        # Already have - don't change
├── data_loader.py              # CREATE - loads index & pagerank
├── tokenizer.py                # CREATE - same as GCP notebook
├── queries_train.json          # Already have
├── data/                       # CREATE - download data here
│   ├── postings_gcp/
│  ✅ Quick Start Checklist

Follow these steps to get your search engine working:

### Phase 1: Setup (1-1.5 hours)
- [ ] Check disk space: ~20 GB free
- [ ] Download `index.pkl` from GCS
- [ ] Download all `.bin` files from GCS (takes time!)
- [ ] Download PageRank `.csv.gz` files
- [ ] Run `verify_data.py` to check downloads

### Phase 2: Code (30-45 minutes)
- [ ] Create `data_loader.py`
- [ ] Create `tokenizer.py`
- [ ] Test tokenizer works
- [ ] Edit `search_frontend.py`:
  - [ ] Add imports and data loading
  - [ ] Implement `/search_body` endpoint

### Phase 3: Test (15 minutes)
- [ ] Install packages: `flask`, `pandas`, `nltk`
- [ ] Run: `python search_frontend.py`
- [ ] Test query: `http://localhost:8080/search?query=python`
- [ ] Run: `python tests\test_search.py`
- [ ] ✅ **Working search engine!**
Optimize (Optional, 1-2 hours)
- [ ] Tune text/PageRank weights
- [ ] Try BM25 instead of TF-IDF
- [ ] Add caching for speed
- [ ] Add other endpoints if required
- [ ] Evaluate with `queries_train.json`es_train.json`
- [ ] Evaluate quality

---

## 🎯 Success Criteria

Your search engine is working when:
- ✅ Server starts without errors
- ✅ Queries return ~100 document IDs
- ✅ Results seem relevant to query
- ✅ Response time < 5 seconds         │
│    - Implement search_body (simple binary)      │
└─────────────────┬───────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────┐
│ 4. Test locally: python search_frontend.py     │
│    Visit: http://localhost:8080/search_body    │
└─────────────────┬───────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────┐
│ 5. IT WORKS! Now improve incrementally:        │
│    - Add TF-IDF scoring                         │
│    - Add PageRank boost                         │
│    - Add title/anchor search                    │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Quick Start Checklist

**To get a working engine in 3-4 hours:**

- [ ] **Choose**: Local (download files) OR Remote (read from GCS)
- [ ] If local: Download index.pkl and posting files from GCS
- [ ] If local: Download PageRank CSV files from GCS
- [ ] If remote: Keep files in GCS, set `WORK_LOCALLY=False`
- [ ] Create `data_loader.py` with load functions
- [ ] Create `tokenizer.py` with same tokenizer as GCP
- [ ] Edit `search_frontend.py`:
  - [ ] Import and load data at startup
  - [ ] Implement `/search_body` with binary scoring
- [ ] Test: `python search_frontend.py`
- [ ] Test query: `http://localhost:8080/search_body?query=python`
- [ ] ✅ **You now have a working search engine!**

---

## 🚀 Next Steps (After Basic Works)

1. **Improve scoring**: Replace binary with TF-IDF
2. **Add PageRank**: Implement `/get_pagerank` endpoint
3. **Combine signals**: Update `/search` to use body + PageRank
4. **Add more endpoints**: `/search_title`, `/search_anchor`
5. **Get doc titles**: Extract from parquet or create mapping
6. **Test with training queries**: Evaluate your MAP score
7. **Deploy**: Upload to GCP or use Colab

---

## ⚠️ Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "File not found" | Check paths in data_loader.py match your downloads |
| "No module named inverted_index_gcp" | Make sure file is in same dire
| Thousands of .bin files | Normal! Each bucket creates multiple files. index.pkl knows where everything is |
| "posting_locs.pickle files?" | Skip them - already merged into index.pkl by your GCP notebook |ctory as search_frontend.py |
| Empty results | Check tokenizeopening and reading .bin files)
- **Subsequent queries**: < 1 second (with caching)
- **Memory usage**: ~2-4 GB (index metadata + cached posting lists)
- **Disk space**: ~10-15 GB (all .bin files with posting data
---

## 📊 Performance Expectations

- **First query**: 2-5 seconds (loading posting lists)
- **Subsequent queries**: < 1 second (with caching)
- **Memory usage**: ~2-4 GB (index + posting lists)
- **Disk space**: ~10-15 GB (all posting files)

---

*Last Updated: December 2024*
*Focus: Get it working first, optimize later!