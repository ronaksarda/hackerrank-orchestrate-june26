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

def strategy_b_process(row, dataset_dir):
    from pipeline.claim_parser import parse_claim
    from pipeline.vision_review import inspect_image
    from pipeline.decision import get_user_risk_flags
    from pipeline.schema import OutputRow, IssueType, ClaimStatus, Severity, normalize_object_part
    
    parsed = parse_claim(row['user_claim'], row['claim_object'])
    
    image_paths = row['image_paths'].split(";")
    visions = []
    for ipath in image_paths:
        full_path = os.path.join(dataset_dir, ipath.strip())
        visions.append(inspect_image(full_path, parsed.claim_object, parsed.object_part, parsed.issue_type, ""))
        
    valid_img = all(v.is_valid_image for v in visions)
    evidence_met = any(v.detected_issue != "none" and v.detected_issue != "unknown" for v in visions)
    
    severities = [v.severity for v in visions if v.severity not in ("unknown", "none")]
    severity = severities[0] if severities else "unknown"
    
    detected_parts = [v.detected_part for v in visions if v.detected_part != "unknown"]
    object_part = detected_parts[0] if detected_parts else parsed.object_part
    
    detected_issues = [v.detected_issue for v in visions if v.detected_issue not in ("unknown", "none")]
    issue_type = detected_issues[0] if detected_issues else parsed.issue_type
    
    risk_flags = get_user_risk_flags(row['user_id'])
    
    try:
        issue_type_enum = IssueType(issue_type)
    except:
        issue_type_enum = IssueType.UNKNOWN
    try:
        sev_enum = Severity(severity)
    except:
        sev_enum = Severity.UNKNOWN
        
    return OutputRow(
        user_id=row['user_id'],
        image_paths=row['image_paths'],
        user_claim=row['user_claim'],
        claim_object=row['claim_object'],
        evidence_standard_met=str(evidence_met).lower(),
        evidence_standard_met_reason="Evaluated by Strategy B",
        risk_flags=";".join(risk_flags) if risk_flags else "none",
        issue_type=issue_type_enum,
        object_part=normalize_object_part(object_part, row['claim_object']),
        claim_status=ClaimStatus.SUPPORTED if evidence_met else ClaimStatus.CONTRADICTED,
        claim_status_justification="Evaluated by Strategy B",
        supporting_image_ids="none",
        valid_image=str(valid_img).lower(),
        severity=sev_enum
    )

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
                    
        print(f"\n--- {strategy_name} Accuracy ---")
        for col in eval_cols:
            acc = correct[col] / total * 100
            print(f"  {col:30s}: {correct[col]}/{total} ({acc:.1f}%)")
        overall = sum(correct.values()) / (len(eval_cols) * total) * 100
        print(f"  Overall field accuracy: {overall:.1f}%")
        print(f"  Runtime: {elapsed:.1f}s")

if __name__ == "__main__":
    evaluate()
