"""
Test tokenizer to ensure it matches GCP notebook behavior.
Run: python tests/test_tokenizer.py
"""

import sys
import os

# Add Backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))

from tokenizer import tokenize

def test_tokenize():
    """Test tokenization with various inputs."""
    
    print("=" * 60)
    print("Testing Tokenizer")
    print("=" * 60)
    
    # Test 1: Basic tokenization
    print("\n[Test 1] Basic tokenization")
    text = "Python programming language"
    tokens = tokenize(text)
    print(f"Input: '{text}'")
    print(f"Output: {tokens}")
    assert tokens == ['python', 'programming', 'language'], "Basic tokenization failed"
    print("✓ PASSED")
    
    # Test 2: Stopword removal
    print("\n[Test 2] Stopword removal")
    text = "This is a test of the tokenizer"
    tokens = tokenize(text)
    print(f"Input: '{text}'")
    print(f"Output: {tokens}")
    # 'this', 'is', 'a', 'of', 'the' should be removed as stopwords
    assert 'this' not in tokens and 'is' not in tokens and 'the' not in tokens
    assert 'test' in tokens and 'tokenizer' in tokens
    print("✓ PASSED")
    
    # Test 3: Corpus stopwords (from GCP notebook)
    print("\n[Test 3] Corpus-specific stopwords")
    text = "category references also external links"
    tokens = tokenize(text)
    print(f"Input: '{text}'")
    print(f"Output: {tokens}")
    # All these should be filtered out
    assert len(tokens) == 0, "Corpus stopwords not removed"
    print("✓ PASSED")
    
    # Test 4: Case insensitivity
    print("\n[Test 4] Case insensitivity")
    text = "Python PYTHON python"
    tokens = tokenize(text)
    print(f"Input: '{text}'")
    print(f"Output: {tokens}")
    assert tokens == ['python', 'python', 'python'], "Case handling failed"
    print("✓ PASSED")
    
    # Test 5: Special characters and numbers
    print("\n[Test 5] Special characters and hashtags")
    text = "machine learning #AI @mention test123"
    tokens = tokenize(text)
    print(f"Input: '{text}'")
    print(f"Output: {tokens}")
    assert 'machine' in tokens and 'learning' in tokens
    assert '#ai' in tokens or 'ai' in tokens  # regex allows #
    assert 'test123' in tokens
    print("✓ PASSED")
    
    # Test 6: Short tokens filtered (less than 2 chars)
    # print("\n[Test 6] Short tokens filtered")
    # text = "a ab abc abcd"
    # tokens = tokenize(text)
    # print(f"Input: '{text}'")
    # print(f"Output: {tokens}")
    # # 'a' should be filtered (stopword + length), 'ab' stays (length=2)
    # assert 'a' not in tokens
    # assert 'ab' in tokens and 'abc' in tokens and 'abcd' in tokens
    # print("✓ PASSED")
    
    # Test 7: Real Wikipedia-style text
    print("\n[Test 7] Wikipedia-style text")
    text = "Python is a high-level programming language. First released in 1991."
    tokens = tokenize(text)
    print(f"Input: '{text}'")
    print(f"Output: {tokens}")
    # Stopwords: is, a, in
    # Corpus stopword: first
    assert 'python' in tokens
    assert 'high-level' in tokens or 'high' in tokens  # hyphenated
    assert 'programming' in tokens
    assert 'language' in tokens
    assert 'released' in tokens
    assert '1991' in tokens
    assert 'first' not in tokens  # corpus stopword
    print("✓ PASSED")
    
    # Test 8: Empty and whitespace
    print("\n[Test 8] Edge cases (empty/whitespace)")
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("the a an") == []  # all stopwords
    print("✓ PASSED")
    
    print("\n" + "=" * 60)
    print("✅ ALL TOKENIZER TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_tokenize()
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)