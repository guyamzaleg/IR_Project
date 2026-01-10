# Test Script Implementation Plan: Precision@10 Evaluation

## Overview
This document outlines the implementation plan for creating a test script that evaluates the search engine's performance using the Precision@10 metric against queries from `queries_train.json`.

---

## 1. Understanding the Input Data

### queries_train.json Structure
- **Format**: JSON dictionary where:
  - **Keys**: Query strings (e.g., "Mount Everest climbing expeditions")
  - **Values**: Arrays of relevant document IDs (ground truth) for each query

### Example Entry
```json
{
  "Mount Everest climbing expeditions": ["47353693", "5208803", "20852640", ...]
}
```

---

## 2. Test Script Architecture

### 2.1 Main Components

#### Component 1: Query Processor
- **Purpose**: Load and parse queries from `queries_train.json`
- **Input**: JSON file path
- **Output**: Dictionary of query strings mapped to ground truth document IDs

#### Component 2: Search Engine Interface
- **Purpose**: Query the search engine and retrieve results
- **Selected Approach**: Import search functions directly from `search_frontend.py`
- **Rationale**: Direct function calls provide efficiency and avoid HTTP overhead

#### Component 3: Precision@10 Calculator
- **Purpose**: Calculate Precision@10 for each query
- **Formula**: 
  ```
  Precision@10 = (Number of relevant documents in top 10 results) / 10
  ```
- **Input**: 
  - Retrieved document IDs (top 10)
  - Ground truth document IDs
- **Output**: Precision score (0.0 to 1.0)

#### Component 4: Results Aggregator and Reporter
- **Purpose**: Collect all metrics and generate comprehensive report
- **Outputs**:
  - Per-query precision scores
  - Average Precision@10 across all queries
  - Statistics (min, max, median)
  - Detailed report with query analysis

---

## 3. Detailed Implementation Steps

### Step 1: Create Test Script File
- **File Name**: `test_precision.py`
- **Location**: `tests/` directory
- **Dependencies**: 
  - `json` (loading queries)
  - `sys` and `os` (path management)
  - Search engine modules from parent directory

### Step 2: Implement Query Loader Function
```python
def load_queries(filepath):
    """
    Load queries and ground truth from JSON file
    
    Args:
        filepath: Path to queries_train.json
    
    Returns:
        dict: {query_text: [relevant_doc_ids]}
    """
```

**Implementation Details**:
- Use `json.load()` to parse file
- Validate file structure
- Handle potential file I/O errors
- Return cleaned data structure

### Step 3: Implement Search Interface Function
```python
def query_search_engine(query_text, top_k=10):
    """
    Query the search engine and return top K results
    
    Args:
        query_text: String query
        top_k: Number of results to retrieve (default: 10)
    
    Returns:
        list: Document IDs of top K results
    """
```

**Implementation Details**:
- Import search function from `search_frontend.py`
- Ensure proper initialization of inverted index and pagerank
- Extract document IDs from search results
- Handle empty results gracefully

### Step 4: Implement Precision@10 Calculator
```python
def calculate_precision_at_k(retrieved_docs, relevant_docs, k=10):
    """
    Calculate Precision@K metric
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of ground truth relevant document IDs
        k: Cut-off rank (default: 10)
    
    Returns:
        float: Precision@K score
    """
```

**Implementation Details**:
- Take first K documents from retrieved results
- Convert both lists to sets for efficient intersection
- Count relevant documents in top K
- Return ratio: relevant_in_topK / K

### Step 5: Implement Main Testing Loop
```python
def run_precision_test(queries_file):
    """
    Main function to run precision tests on all queries
    
    Args:
        queries_file: Path to queries_train.json
    
    Returns:
        dict: Results for each query with precision scores
    """
```

**Implementation Details**:
- Load queries using Step 2 function
- Initialize search engine components
- For each query:
  - Query the search engine
  - Calculate Precision@10
  - Store results
- Calculate aggregate statistics
- Return complete results dictionary

### Step 6: Implement Results Reporter
```python
def generate_report(results):
    """
    Generate comprehensive test report
    
    Args:
        results: Dictionary of test results
    
    Outputs:
        - Console summary
        - Detailed CSV file
        - JSON results file
    """
```

**Report Should Include**:
1. **Summary Statistics**:
   - Mean Precision@10
   - Median Precision@10
   - Standard deviation
   - Min and Max scores
   
2. **Per-Query Results**:
   - Query text
   - Precision@10 score
   - Number of relevant documents found
   - Total ground truth documents
   - **Article IDs that were hits** (matched with ground truth)
   
3. **Performance Analysis**:
   - Queries with perfect precision (1.0)
   - Queries with zero precision (0.0)
   - Distribution histogram

### Step 7: Create Output Files
- **CSV File**: `precision_results.csv`
  ```
  Query,Precision@10,Relevant_Found,Total_Relevant
  "Mount Everest...",0.7,7,46
  ```
  
- **JSON File**: `precision_results.json`
  ```json
  {
    "summary": {
      "mean_precision": 0.65,
      "queries_tested": 30
    },
    "per_query": {...}
  }
  ```

---

## 4. Code Structure Template

### File: `tests/test_precision.py`

```python
#!/usr/bin/env python3
"""
Test script for evaluating search engine performance using Precision@10
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import search engine components
from search_frontend import (function_name)
from Backend.data_Loader import load_index, load_pagerank

# ============== CONFIGURATION ==============
QUERIES_FILE = "queries_train.json"
OUTPUT_DIR = "tests/results"
K = 10  # For Precision@10

# ============== IMPLEMENTATION ==============

def load_queries(filepath):
    # TODO: Implement
    pass

def query_search_engine(query_text, top_k=10):
    # TODO: Implement
    pass

def calculate_precision_at_k(retrieved_docs, relevant_docs, k=10):
    # TODO: Implement
    pass

def run_precision_test(queries_file):
    # TODO: Implement
    pass

def generate_report(results):
    # TODO: Implement
    pass

def main():
    print("="*60)
    print("SEARCH ENGINE PRECISION@10 EVALUATION")
    print("="*60)
    
    # Run tests
    results = run_precision_test(QUERIES_FILE)
    
    # Generate report
    generate_report(results)
    
    print("\n✓ Testing complete!")

if __name__ == "__main__":
    main()
```

---

## 5. Testing Considerations

### 5.1 Edge Cases to Handle
1. **Empty Results**: Query returns no results
2. **Missing Ground Truth**: Query not in JSON file
3. **ID Format Mismatch**: String vs integer document IDs
4. **Fewer Than 10 Results**: Search returns < 10 documents

### 5.2 Validation Checks
- Verify all queries in JSON are tested
- Ensure document ID formats match between systems
- Check for duplicate document IDs in results
- Validate that Precision@10 is always between 0.0 and 1.0

---

## 6. Expected Outputs

### Console Output Example
```
==============================================================
SEARCH ENGINE PRECISION@10 EVALUATION
==============================================================

Loading queries from queries_train.json...
✓ Loaded 30 queries

Initializing search engine...
✓ Search engine ready

Testing queries:
[1/30] Mount Everest climbing expeditions: P@10 = 0.70
[2/30] Great Fire of London 1666: P@10 = 0.50
[3/30] Nanotechnology materials science: P@10 = 0.80
...

==============================================================
RESULTS SUMMARY
==============================================================
Total Queries Tested: 30
Mean Precision@10: 0.653
Median Precision@10: 0.700
Std Deviation: 0.182
Min Precision@10: 0.100
Max Precision@10: 1.000

Perfect Scores (P@10 = 1.0): 5 queries
Zero Scores (P@10 = 0.0): 1 query

✓ Testing complete!
Results saved to:
  - tests/results/precision_results.csv
  - tests/results/precision_results.json
```

---

## 7. Integration with Existing Code

### Required Modifications to search_frontend.py
- **Option 1**: Export search function that returns document IDs
  ```python
  def search_and_return_ids(query, top_k=100):
      # Existing search logic
      # Return list of doc IDs
      return [doc['id'] for doc in results]
  ```

- **Option 2**: Modify existing `/search` endpoint to accept parameter
  ```python
  @app.route("/search")
  def search():
      query = request.args.get('query', '')
      return_format = request.args.get('format', 'json')
      # Return IDs only if format='ids'
  ```

### Recommendation
Implement Option 1 for cleaner separation and direct function calls.

---

## 8. Performance Optimization

### For Large Query Sets
1. **Batch Processing**: Process queries in batches
2. **Caching**: Cache loaded index and pagerank
3. **Parallel Processing**: Use multiprocessing for independent queries
4. **Progress Tracking**: Add progress bar (e.g., `tqdm`)

### Example with Progress Bar
```python
from tqdm import tqdm

for query, relevant_docs in tqdm(queries.items(), desc="Testing"):
    # Process query
    pass
```

---

## 9. Future Enhancements

### Additional Metrics to Consider
1. **Precision@5**: More strict top-k evaluation
2. **Recall@10**: Measure coverage of relevant documents
3. **MAP (Mean Average Precision)**: Consider ranking order
4. **NDCG@10**: Consider graded relevance
5. **MRR (Mean Reciprocal Rank)**: Find first relevant result

### Visualization Options
1. Precision distribution histogram
2. Query difficulty analysis
3. Performance comparison across query categories
4. Correlation with query length

---

## 10. Implementation Checklist

- [ ] Create `tests/test_precision.py` file
- [ ] Implement `load_queries()` function
- [ ] Implement `query_search_engine()` function
- [ ] Implement `calculate_precision_at_k()` function
- [ ] Implement `run_precision_test()` function
- [ ] Implement `generate_report()` function
- [ ] Add error handling and logging
- [ ] Create output directory structure
- [ ] Test with subset of queries first
- [ ] Run full test suite
- [ ] Generate and review reports
- [ ] Document results in README

---

## 11. Dependencies

### Python Packages Required
```
json (standard library)
sys (standard library)
os (standard library)
pathlib (standard library)
statistics (standard library)
csv (standard library)
```

### Optional Dependencies
```
tqdm (progress bars)
pandas (advanced data analysis)
matplotlib (visualizations)
numpy (statistical calculations)
```

---

## 12. Timeline Estimate

| Task | Estimated Time |
|------|----------------|
| Setup and boilerplate | 30 minutes |
| Query loader implementation | 20 minutes |
| Search interface implementation | 30 minutes |
| Precision calculator | 20 minutes |
| Main testing loop | 30 minutes |
| Report generation | 40 minutes |
| Testing and debugging | 1 hour |
| Documentation | 20 minutes |
| **Total** | **~3.5 hours** |

---

## Conclusion

This plan provides a comprehensive roadmap for implementing a robust test script to evaluate search engine performance using Precision@10. The modular design allows for easy extension to additional metrics and ensures maintainability. Follow the steps sequentially, test incrementally, and document findings for future reference.
