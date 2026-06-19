import pandas as pd
import os
import sys
import time

# Add code/ to path so we can import pipeline
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code_dir)
os.chdir(code_dir)

from dotenv import load_dotenv
load_dotenv()

from main import process_row as strategy_a_process
from pipeline.decision import load_static_data



def evaluate():
    dataset_dir = os.path.join(code_dir, "..", "dataset")
    dataset_dir = os.path.abspath(dataset_dir)
    sample_path = os.path.join(dataset_dir, "sample_claims.csv")
    
    print(f"Loading static data from {dataset_dir}...")
    load_static_data(dataset_dir)
    
    print(f"Reading sample claims from {sample_path}...")
    df = pd.read_csv(sample_path)
    total = len(df)
    
    eval_cols = [
        "claim_status", "issue_type", "object_part", 
        "severity", "valid_image", "evidence_standard_met"
    ]
    
    if os.path.exists("mismatches.txt"):
        os.remove("mismatches.txt")
    
    for strategy_name, strategy_fn in [("Strategy A (Single-shot Analyzer)", strategy_a_process)]:
        predictions = []
        start_time = time.time()
        
        print(f"\nEvaluating {strategy_name}...")
        for idx, row in df.iterrows():
            try:
                out_row = strategy_fn(row, dataset_dir)
                pred = out_row.model_dump(by_alias=True)
                predictions.append(pred)
            except Exception as e:
                print(f"  ERROR on row {idx+1}: {e}")
                predictions.append({col: "unknown" for col in eval_cols})
        
        elapsed = time.time() - start_time
        correct = {col: 0 for col in eval_cols}
        
        for i in range(total):
            gt = df.iloc[i]
            pred = predictions[i]
            for col in eval_cols:
                gt_val = str(gt[col]).strip().lower()
                pred_val = str(pred.get(col, "")).strip().lower()
                if gt_val == pred_val:
                    correct[col] += 1
                elif col in ["issue_type", "severity"]:
                    with open("mismatches.txt", "a") as f:
                        f.write(f"USER: {gt['user_id']} | CLAIM: {gt['user_claim']}\n")
                        f.write(f"FIELD: {col} | GT: {gt_val} | PRED: {pred_val}\n")
                        f.write(f"JUSTIFICATION: {pred.get('claim_status_justification', '')}\n")
                        f.write("-" * 40 + "\n")
                    
        print(f"\n--- {strategy_name} Accuracy ---")
        for col in eval_cols:
            acc = correct[col] / total * 100
            print(f"  {col:30s}: {correct[col]}/{total} ({acc:.1f}%)")
        overall = sum(correct.values()) / (len(eval_cols) * total) * 100
        print(f"  Overall field accuracy: {overall:.1f}%")
        print(f"  Runtime: {elapsed:.1f}s")

if __name__ == "__main__":
    evaluate()
