import hashlib
import re
import zlib
from typing import Dict, Tuple

import tiktoken
from pdfminer.high_level import extract_text

# A simple list of stopwords
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same",
    "she", "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours", "yourself", "yourselves"
])

# Heuristics for PII and IP risk detection
PII_PATTERNS = {
    "EMAIL": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    "PHONE": re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    "IP_ADDRESS": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
}
IP_RISK_PATTERNS = {
    "QUOTES": re.compile(r'["\'](.*?)["\']'),
}


def read_file_content(file_path: str) -> str:
    """Reads content from a file, supporting .txt, .md, and .pdf."""
    if file_path.endswith(".pdf"):
        return extract_text(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def get_file_sha256(file_path: str) -> str:
    """Computes the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def analyze_text_metrics(text: str, token_model: str) -> Dict:
    """Analyzes various metrics of the text."""
    char_count = len(text)
    word_count = len(text.split())

    try:
        enc = tiktoken.encoding_for_model(token_model)
        token_count = len(enc.encode(text))
    except Exception:
        token_count = len(text.split())  # Fallback to word count

    words = re.findall(r'\w+', text.lower())
    non_stopwords = [word for word in words if word not in STOPWORDS]
    semantic_density = len(non_stopwords) / len(words) if words else 0

    compressed_text = zlib.compress(text.encode("utf-8"))
    compression_ratio = len(compressed_text) / char_count if char_count else 0

    return {
        "token_count": token_count,
        "word_count": word_count,
        "char_count": char_count,
        "semantic_density": semantic_density,
        "compression_ratio": compression_ratio,
    }


def analyze_risk(text: str) -> Dict:
    """Analyzes the text for PII and IP risks."""
    pii_summary = {key: len(pattern.findall(text)) for key, pattern in PII_PATTERNS.items()}
    ip_risk_summary = {key: len(pattern.findall(text)) for key, pattern in IP_RISK_PATTERNS.items()}

    pii_detected = any(pii_summary.values())
    ip_risk_detected = any(ip_risk_summary.values())

    return {
        "pii_detected": pii_detected,
        "ip_risk_detected": ip_risk_detected,
        "pii_summary": {k: v for k, v in pii_summary.items() if v > 0},
        "ip_risk_summary": {k: v for k, v in ip_risk_summary.items() if v > 0},
    }


def get_extractive_summary(text: str, num_sentences: int = 3) -> str:
    """Generates an extractive summary from the text."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # A simple scoring mechanism: score sentences based on length
    scored_sentences = sorted(sentences, key=len, reverse=True)
    summary_sentences = sorted(scored_sentences[:num_sentences], key=text.find)
    return " ".join(summary_sentences)