import os
import hashlib
import pytest
from local_analyzer import (
    read_file_content,
    get_file_sha256,
    analyze_text_metrics,
    analyze_risk,
    get_extractive_summary,
)

TEST_FILE_CONTENT = """This is a test.
Email me at test@example.com or call (123) 456-7890.
"This is a quote."
"""
TEST_FILE_PATH = "test/dummy_test_file.txt"

@pytest.fixture
def dummy_test_file():
    """Creates a dummy test file and returns its path."""
    with open(TEST_FILE_PATH, "w") as f:
        f.write(TEST_FILE_CONTENT)
    yield TEST_FILE_PATH
    os.remove(TEST_FILE_PATH)


def test_read_file_content(dummy_test_file):
    content = read_file_content(dummy_test_file)
    assert content == TEST_FILE_CONTENT


def test_get_file_sha256(dummy_test_file):
    sha256 = get_file_sha256(dummy_test_file)
    hasher = hashlib.sha256()
    hasher.update(TEST_FILE_CONTENT.encode('utf-8'))
    expected_sha256 = hasher.hexdigest()
    assert sha256 == expected_sha256


def test_analyze_text_metrics():
    metrics = analyze_text_metrics(TEST_FILE_CONTENT, "gpt-4o-mini")
    assert metrics["token_count"] > 0
    assert metrics["word_count"] > 0
    assert metrics["char_count"] > 0
    assert 0 <= metrics["semantic_density"] <= 1
    assert 0 <= metrics["compression_ratio"] <= 1


def test_analyze_risk():
    risk = analyze_risk(TEST_FILE_CONTENT)
    assert risk["pii_detected"] is True
    assert risk["ip_risk_detected"] is True
    assert "EMAIL" in risk["pii_summary"]
    assert "PHONE" in risk["pii_summary"]
    assert "QUOTES" in risk["ip_risk_summary"]


def test_get_extractive_summary():
    summary = get_extractive_summary(TEST_FILE_CONTENT)
    assert 'Email me at' in summary
    assert 'This is a quote' in summary