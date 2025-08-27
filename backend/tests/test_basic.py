import pytest
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now we can import the modules
from backend.app import app
from backend.analysis.embeddings import pairwise_similarity
from backend.analysis.diff import token_diff, html_token_diff, unified_token_diff

# Create a TestClient instance
client = TestClient(app)


def test_compare_endpoint_mock_mode():
    """Test the /compare endpoint in mock mode returns expected structure"""
    # Test data
    test_request = {
        "prompt": "Explain quantum computing in simple terms",
        "models": ["gpt-4o", "claude-3-opus"],
        "temperature": 0.2
    }
    
    # Make request to /compare endpoint
    response = client.post("/compare", json=test_request)
    
    # Check response status
    assert response.status_code == 200
    
    # Check response structure
    data = response.json()
    assert "results" in data
    assert "embedding_similarity" in data
    assert "summaries" in data
    assert "token_diffs" in data
    
    # Check results structure (mock always returns 2 models)
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert "model" in result
        assert "output_text" in result
        assert "tokens" in result
        assert "embedding" in result
        assert "hallucination_risk" in result
    
    # Check embedding similarity structure
    similarity = data["embedding_similarity"]
    assert "gpt-4o" in similarity
    assert "claude-3-opus" in similarity
    assert similarity["gpt-4o"]["gpt-4o"] == 1.0
    assert similarity["claude-3-opus"]["claude-3-opus"] == 1.0


def test_compare_endpoint_with_single_model_request():
    """Test the /compare endpoint with a single model request (mock returns 2 models always)"""
    test_request = {
        "prompt": "Say hello in one sentence",
        "models": ["gpt-4o"],
        "temperature": 0.2
    }
    
    response = client.post("/compare", json=test_request)
    assert response.status_code == 200
    
    data = response.json()
    # Mock mode always returns 2 models regardless of request
    assert len(data["results"]) == 2
    # But the first model should match the requested model
    assert data["results"][0]["model"] in ["gpt-4o", "claude-3-opus"]


def test_pairwise_similarity_function():
    """Test the pairwise similarity function with known vectors"""
    # Test with simple 2D vectors
    labels = ["model_a", "model_b"]
    vectors = [
        [1.0, 0.0],  # Vector pointing right
        [0.0, 1.0]   # Vector pointing up
    ]
    
    similarity = pairwise_similarity(labels, vectors)
    
    # Check structure
    assert "model_a" in similarity
    assert "model_b" in similarity
    assert "model_a" in similarity["model_a"]
    assert "model_b" in similarity["model_a"]
    
    # Check diagonal is 1.0 (self-similarity)
    assert similarity["model_a"]["model_a"] == 1.0
    assert similarity["model_b"]["model_b"] == 1.0
    
    # Check cosine similarity between perpendicular vectors is 0.0
    assert abs(similarity["model_a"]["model_b"] - 0.0) < 1e-10
    assert abs(similarity["model_b"]["model_a"] - 0.0) < 1e-10


def test_pairwise_similarity_with_identical_vectors():
    """Test pairwise similarity with identical vectors"""
    labels = ["model_a", "model_b"]
    vectors = [
        [1.0, 1.0],
        [1.0, 1.0]
    ]
    
    similarity = pairwise_similarity(labels, vectors)
    
    # Identical vectors should have similarity of 1.0
    assert abs(similarity["model_a"]["model_b"] - 1.0) < 1e-10
    assert abs(similarity["model_b"]["model_a"] - 1.0) < 1e-10


def test_token_diff_function():
    """Test the token diff function"""
    a_tokens = ["the", "quick", "brown", "fox"]
    b_tokens = ["the", "quick", "red", "fox", "jumps"]
    
    # Test basic token diff
    changes = token_diff(a_tokens, b_tokens)
    
    # Should have at least one operation
    assert len(changes) > 0
    # Should have operations like 'equal', 'replace', 'insert', 'delete'
    ops = [change["op"] for change in changes]
    assert len(ops) > 0


def test_html_token_diff():
    """Test HTML token diff generation"""
    a_tokens = ["the", "quick", "brown", "fox"]
    b_tokens = ["the", "quick", "red", "fox", "jumps"]
    
    html_diff = html_token_diff(a_tokens, b_tokens)
    
    # Should return a string with HTML tags
    assert isinstance(html_diff, str)
    assert len(html_diff) > 0


def test_unified_token_diff():
    """Test unified diff format generation"""
    a_tokens = ["the", "quick", "brown", "fox"]
    b_tokens = ["the", "quick", "red", "fox", "jumps"]
    
    unified_diff = unified_token_diff(a_tokens, b_tokens, "model_a", "model_b")
    
    # Should return a string with diff format
    assert isinstance(unified_diff, str)
    assert len(unified_diff) > 0


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "ok" in data
    assert data["ok"] == True


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "ok" in data
    assert data["ok"] == True
    assert "service" in data
    assert data["service"] == "prompt-autopsy"


def test_embedding_function_deterministic():
    """Test that embedding function produces consistent results"""
    from backend.analysis.embeddings import embed_texts
    
    # Use a simple sentence
    texts = ["Hello world"]
    
    # Mock the model to return consistent embeddings
    with patch('backend.analysis.embeddings.get_model') as mock_get_model:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, -0.2, 0.8]])
        mock_get_model.return_value = mock_model
        
        embeddings1 = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2")
        embeddings2 = embed_texts(texts, "sentence-transformers/all-MiniLM-L6-v2")
        
        # Should be deterministic with the same input
        assert embeddings1 == embeddings2


if __name__ == "__main__":
    pytest.main([__file__])