import os
from unittest.mock import patch, Mock
import requests
from api_analyzer import get_abstractive_summary, calculate_cost

# Mock config
CONFIG = {
    "OPENAI_API_KEY": "test_key",
    "HF_API_TOKEN": "test_key",
    "TOKEN_MODEL": "gpt-4o-mini",
    "MAX_SUMMARY_TOKENS": 150,
    "INPUT_COST_PER_1K": 0.030,
    "OUTPUT_COST_PER_1K": 0.060,
}

@patch("requests.post")
def test_get_abstractive_summary_openai(mock_post):
    """Test OpenAI summary generation."""
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OpenAI summary."}}]
    }
    mock_post.return_value = mock_response

    summary, provider, model = get_abstractive_summary("Some text", CONFIG)
    assert summary == "OpenAI summary."
    assert provider == "openai"
    assert model == "gpt-4o-mini"

@patch("requests.post")
def test_get_abstractive_summary_hf_only(mock_post):
    """Test Hugging Face summary generation when only HF token is present."""
    mock_hf_success = Mock()
    mock_hf_success.raise_for_status.return_value = None
    mock_hf_success.json.return_value = [{"summary_text": "HF summary."}]
    mock_post.return_value = mock_hf_success

    config_hf_only = {"HF_API_TOKEN": "test_key"}

    summary, provider, model = get_abstractive_summary("Some text", config_hf_only)
    assert summary == "HF summary."
    assert provider == "huggingface"
    assert model == "sshleifer/distilbart-cnn-12-6"

@patch("requests.post")
def test_get_abstractive_summary_fallback(mock_post):
    """Test fallback to local summary."""
    mock_post.side_effect = requests.RequestException

    config_no_keys = {}
    summary, provider, model = get_abstractive_summary("This is a test sentence.", config_no_keys)
    assert summary == "This is a test sentence."  # Extractive summary will return the longest sentence
    assert provider == "local"
    assert model == "extractive"


def test_calculate_cost():
    """Test cost calculation."""
    cost = calculate_cost(1000, 200, CONFIG)
    assert cost == (1 * 0.030) + (0.2 * 0.060)