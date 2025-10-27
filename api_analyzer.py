import os
import json
import requests
from typing import Dict, Optional, Tuple

def get_abstractive_summary(text: str, config: Dict) -> Tuple[str, str, Optional[str]]:
    """
    Gets an abstractive summary from OpenAI or Hugging Face, with a local fallback.
    Returns (summary, provider, model_used).
    """
    if config.get("OPENAI_API_KEY"):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {config['OPENAI_API_KEY']}"},
                json={
                    "model": config.get("TOKEN_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                        {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
                    ],
                    "max_tokens": config.get("MAX_SUMMARY_TOKENS", 200),
                },
            )
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"]
            return summary, "openai", config.get("TOKEN_MODEL", "gpt-4o-mini")
        except requests.RequestException as e:
            print(f"OpenAI API request failed: {e}")

    if config.get("HF_API_TOKEN"):
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6",
                headers={"Authorization": f"Bearer {config['HF_API_TOKEN']}"},
                json={"inputs": text},
            )
            response.raise_for_status()
            summary = response.json()[0]["summary_text"]
            return summary, "huggingface", "sshleifer/distilbart-cnn-12-6"
        except requests.RequestException as e:
            print(f"Hugging Face API request failed: {e}")

    # Fallback to local extractive summary if both APIs fail
    from local_analyzer import get_extractive_summary
    return get_extractive_summary(text), "local", "extractive"


def get_web_similarity(text: str, config: Dict) -> float:
    """
    Performs a web similarity check using Bing or Google.
    Returns a score from 0 to 1.
    """
    # This is a placeholder for the actual implementation
    return 0.0

def calculate_cost(token_count: int, summary_token_count: int, config: Dict) -> float:
    """Calculates the estimated cost of the analysis in USD."""
    input_cost = (token_count / 1000) * config.get("INPUT_COST_PER_1K", 0.0)
    output_cost = (summary_token_count / 1000) * config.get("OUTPUT_COST_PER_1K", 0.0)
    return input_cost + output_cost