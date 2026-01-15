import json
import time
import re
import numpy as np
from pathlib import Path
import gensim.downloader as api

# Import tokenization settings from indexer
import sys
sys.path.append('..')
from Backend.tokenizer import RE_WORD

# Stopwords
english_stopwords = frozenset([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
    'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)

def tokenize_query(query, re_word, stopwords):
    """Tokenize and clean a query."""
    tokens = [m.group() for m in re_word.finditer(query.lower())]
    return [t for t in tokens if t not in stopwords]

def generate_embedding(tokens, model):
    """Generate embedding for a list of tokens."""
    # Filter tokens that exist in the model
    valid_tokens = [t for t in tokens if t in model]
    
    if not valid_tokens:
        return None
    
    # Use get_mean_vector for efficiency
    vec = model.get_mean_vector(valid_tokens, pre_normalize=False)
    return vec.astype('float32')

def test_embedding_speed(model_name='glove-wiki-gigaword-100'):
    """Test the speed of embedding generation on training queries."""
    
    print(f"{'='*70}")
    print(f"Testing Embedding Speed with {model_name}")
    print(f"{'='*70}\n")
    
    # Load training queries
    print("Loading training queries...")
    queries_path = Path(__file__).parent / 'queries_train.json'
    with open(queries_path, 'r') as f:
        queries_data = json.load(f)
    
    queries = list(queries_data.keys())
    print(f"Loaded {len(queries)} queries\n")
    
    # Load pre-trained model
    print(f"Loading model: {model_name}...")
    start_load = time.time()
    model = api.load(model_name)
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    print(f"   Model size: {len(model)} words\n")
    
       # Test tokenization speed
    print("Testing tokenization...")
    start_tokenize = time.perf_counter()  # Use perf_counter for better precision
    tokenized_queries = []
    for query in queries:
        tokens = tokenize_query(query, RE_WORD, all_stopwords)
        tokenized_queries.append(tokens)
    tokenize_time = time.perf_counter() - start_tokenize
    
    avg_tokens = np.mean([len(t) for t in tokenized_queries])
    print(f"✅ Tokenized {len(queries)} queries in {tokenize_time:.6f} seconds")  # More precision
    print(f"   Average tokens per query: {avg_tokens:.1f}")
    if tokenize_time > 0:
        print(f"   Speed: {len(queries)/tokenize_time:.0f} queries/second\n")
    else:
        print(f"   Speed: Too fast to measure (< 1μs per query)\n")
    
    # Test embedding generation speed
    print("Testing embedding generation...")
    start_embed = time.perf_counter()  # Use perf_counter here too
    embeddings = []
    successful = 0
    failed = 0
    
    for tokens in tokenized_queries:
        emb = generate_embedding(tokens, model)
        if emb is not None:
            embeddings.append(emb)
            successful += 1
        else:
            failed += 1
    
    embed_time = time.perf_counter() - start_embed
    
    print(f"✅ Generated embeddings in {embed_time:.6f} seconds")  # More precision
    print(f"   Successful: {successful}/{len(queries)}")
    print(f"   Failed (no valid tokens): {failed}")
    if embed_time > 0:
        print(f"   Speed: {successful/embed_time:.0f} embeddings/second")
    print(f"   Embedding dimension: {embeddings[0].shape[0] if embeddings else 'N/A'}\n")
    
    # Total time
    total_time = tokenize_time + embed_time
    print(f"{'='*70}")
    print(f"Summary:")
    print(f"  Model loading time:    {load_time:.2f}s")
    print(f"  Tokenization time:     {tokenize_time:.6f}s")  # More precision
    print(f"  Embedding time:        {embed_time:.6f}s")     # More precision
    print(f"  Total processing time: {total_time:.6f}s")     # More precision
    if total_time > 0:
        print(f"  Overall speed:         {successful/total_time:.0f} queries/second")
    print(f"{'='*70}\n")
    
    # Show some examples
    print("Example tokenized queries:")
    for i, (query, tokens) in enumerate(zip(queries[:3], tokenized_queries[:3])):
        print(f"  {i+1}. '{query}'")
        print(f"     Tokens: {tokens}")
        if i < len(embeddings):
            print(f"     Embedding shape: {embeddings[i].shape}")
        print()
    
    return {
        'model_name': model_name,
        'num_queries': len(queries),
        'load_time': load_time,
        'tokenize_time': tokenize_time,
        'embed_time': embed_time,
        'total_time': total_time,
        'successful': successful,
        'failed': failed,
        'speed': successful/total_time
    }

def compare_models():
    """Compare speed across different model sizes."""
    models = [
        'glove-wiki-gigaword-50',
        'glove-wiki-gigaword-100',
        'glove-wiki-gigaword-200',
    ]
    
    results = []
    for model_name in models:
        try:
            result = test_embedding_speed(model_name)
            results.append(result)
        except Exception as e:
            print(f"Error with {model_name}: {e}\n")
    
    # Print comparison table
    if results:
        print(f"\n{'='*70}")
        print("Model Comparison:")
        print(f"{'='*70}")
        print(f"{'Model':<30} {'Load (s)':<12} {'Embed (s)':<12} {'Speed (q/s)':<12}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['model_name']:<30} {r['load_time']:<12.2f} {r['embed_time']:<12.4f} {r['speed']:<12.0f}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    # Test with default model
    test_embedding_speed('glove-wiki-gigaword-100')
    
    # Uncomment to compare multiple models
    # compare_models()
