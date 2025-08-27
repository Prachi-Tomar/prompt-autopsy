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


def test_compare_mock_mode():
    """Ensure mock mode works when no API keys are present."""
    # Unset environment variables to simulate missing API keys
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "GOOGLE_CLOUD_PROJECT": "",
        "VERTEX_LOCATION": ""
    }, clear=True):
        response = client.post("/compare", json={
            "prompt": "Hello world",
            "models": ["mock-model-a", "mock-model-b"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "output_text" in data["results"][0]


def test_embedding_similarity():
    from backend.analysis.embeddings import cosine_similarity
    import numpy as np
    vec1 = np.array([[0.1, 0.2, 0.3]])
    vec2 = np.array([[0.1, 0.2, 0.25]])
    score = cosine_similarity(vec1, vec2)
    assert isinstance(score, np.ndarray)
    assert 0.0 <= score[0][0] <= 1.0


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