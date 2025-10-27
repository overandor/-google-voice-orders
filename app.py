import os
import sys
import time
import argparse
import json
import logging
import streamlit as st
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler
from schemas.report_schema import Report, TextMetrics, RiskFlags, AnalysisResult, Provenance
from local_analyzer import read_file_content, get_file_sha256, analyze_text_metrics, analyze_risk, get_extractive_summary
from api_analyzer import get_abstractive_summary, calculate_cost

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "HF_API_TOKEN": os.getenv("HF_API_TOKEN"),
    "BING_API_KEY": os.getenv("BING_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GOOGLE_CSE_ID": os.getenv("GOOGLE_CSE_ID"),
    "TOKEN_MODEL": os.getenv("TOKEN_MODEL", "gpt-4o-mini"),
    "INPUT_COST_PER_1K": float(os.getenv("INPUT_COST_PER_1K", 0.030)),
    "OUTPUT_COST_PER_1K": float(os.getenv("OUTPUT_COST_PER_1K", 0.060)),
    "MAX_SUMMARY_TOKENS": int(os.getenv("MAX_SUMMARY_TOKENS", 200)),
}

# --- Neomorphic UI Styling ---
def setup_streamlit_ui():
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
    body {
        background-color: #ECECEC;
    }
    .stApp {
        background-color: #ECECEC;
    }
    .card {
        background-color: #ECECEC;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 20px 20px 60px #c8c8c8, -20px -20px 60px #ffffff;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def process_file(file_path: str):
    """Processes a single file and returns a report."""
    logging.info(f"Processing file: {file_path}")
    content = read_file_content(file_path)
    sha256 = get_file_sha256(file_path)

    # Local Analysis
    text_metrics_data = analyze_text_metrics(content, CONFIG["TOKEN_MODEL"])
    risk_data = analyze_risk(content)
    extractive_summary = get_extractive_summary(content)

    # API Analysis
    abstractive_summary, provider, model = get_abstractive_summary(content, CONFIG)

    # Placeholder for influence score and cost
    influence_score = 0.0  # To be implemented
    summary_token_count = len(abstractive_summary.split()) # Fallback
    cost = calculate_cost(text_metrics_data["token_count"], summary_token_count, CONFIG)

    # Create Report
    report = Report(
        file_path=file_path,
        metrics=TextMetrics(**text_metrics_data),
        risk=RiskFlags(**risk_data),
        analysis=AnalysisResult(
            provider_used=provider,
            model_used=model,
            extractive_summary=extractive_summary,
            abstractive_summary=abstractive_summary,
            influence_score=influence_score,
            cost_usd=cost,
        ),
        provenance=Provenance(
            file_sha256=sha256,
            config_snapshot=CONFIG,
        ),
    )
    return report

def run_cli(file_path: str):
    """Runs the application in CLI mode for a single file."""
    report = process_file(file_path)
    print(report.model_dump_json(indent=2))

def run_streamlit():
    """Runs the Streamlit UI."""
    setup_streamlit_ui()
    st.title("Voyager Intelligence Engine")

    input_dir = "inputs"
    if not os.path.exists(input_dir):
        st.error(f"Input directory not found: {input_dir}")
        return

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            st.markdown(f'<div class="card"><h3>Analysis for: {filename}</h3></div>', unsafe_allow_html=True)
            with st.spinner(f"Processing {filename}..."):
                report = process_file(file_path)

                # Save report
                output_path = f"outputs/report_{report.provenance.report_uuid}.json"
                with open(output_path, "w") as f:
                    f.write(report.model_dump_json(indent=2))

                # Display report
                st.json(report.model_dump())

def main():
    parser = argparse.ArgumentParser(description="Voyager Intelligence Engine")
    parser.add_argument("--file", help="Path to a single file to process in CLI mode.")
    parser.add_argument("--watch", action="store_true", help="Watch the inputs directory for new files.")
    args = parser.parse_args()

    if args.file:
        run_cli(args.file)
    elif args.watch:
        input_dir = "inputs"
        logging.info(f"Watching directory: {input_dir}")
        event_handler = LoggingEventHandler()
        observer = Observer()
        observer.schedule(event_handler, input_dir, recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    else:
        run_streamlit()

if __name__ == "__main__":
    # Self-test mode
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_cli('test/hello.txt')
    else:
        main()