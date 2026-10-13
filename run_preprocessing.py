"""
UrbanPulse - Multimodal Preprocessing Runner
Executes the full preprocessing and deduplication pipeline across all 51,000 raw multimodal observations.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.pipeline import MultimodalPreprocessor

def main():
    print("=" * 80)
    print("URBANPULSE MULTIMODAL PREPROCESSING ENGINE - PHASE 2")
    print(f"Dataset Source: {BASE_DIR}")
    print("=" * 80)

    output_dir = os.path.join(BASE_DIR, "cleaned_output")
    preprocessor = MultimodalPreprocessor(dataset_dir=BASE_DIR, output_dir=output_dir)
    results = preprocessor.run_full_pipeline()

    print("\n" + "=" * 80)
    print("PREPROCESSING EXECUTION METRICS:")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    print(f"\n[+] Cleaned artifacts generated in: {output_dir}")

if __name__ == "__main__":
    main()
