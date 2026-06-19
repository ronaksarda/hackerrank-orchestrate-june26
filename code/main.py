"""
Multi-Modal Evidence Review System — Entry Point
"""
import pandas as pd
import argparse
import os
import time
from dotenv import load_dotenv
load_dotenv()

from pipeline.schema import OutputRow, IssueType, ClaimStatus, Severity, normalize_object_part
from pipeline.analyzer import analyze_claim
from pipeline.decision import load_static_data, get_evidence_req_desc, get_user_history_context
import concurrent.futures

def process_row(row, dataset_dir) -> OutputRow:
    user_id = row['user_id']
    image_paths_raw = row['image_paths']
    user_claim = row['user_claim']
    claim_object = row['claim_object']

    user_history = get_user_history_context(user_id)
    req_desc = get_evidence_req_desc(claim_object, "general")

    result = analyze_claim(
        user_id=user_id,
        image_paths_raw=image_paths_raw,
        user_claim=user_claim,
        claim_object=claim_object,
        dataset_dir=dataset_dir,
        evidence_req_desc=req_desc,
        user_history=user_history
    )

    try:
        issue_type = IssueType(result.get("issue_type", "unknown"))
    except ValueError:
        issue_type = IssueType.UNKNOWN

    try:
        claim_status = ClaimStatus(result.get("claim_status", "not_enough_information"))
    except ValueError:
        claim_status = ClaimStatus.NOT_ENOUGH_INFORMATION

    try:
        severity = Severity(result.get("severity", "unknown"))
    except ValueError:
        severity = Severity.UNKNOWN

    object_part = normalize_object_part(result.get("object_part", "unknown"), claim_object)

    return OutputRow(
        user_id=user_id,
        image_paths=image_paths_raw,
        user_claim=user_claim,
        claim_object=claim_object,
        evidence_standard_met=str(result.get("evidence_standard_met", "false")).lower(),
        evidence_standard_met_reason=result.get("evidence_standard_met_reason", "Unable to evaluate."),
        risk_flags=result.get("risk_flags", "none"),
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=result.get("claim_status_justification", "Unable to evaluate."),
        supporting_image_ids=result.get("supporting_image_ids", "none"),
        valid_image=str(result.get("valid_image", "true")).lower(),
        severity=severity
    )

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review System")
    parser.add_argument("--input", type=str, default="../dataset/claims.csv")
    parser.add_argument("--output", type=str, default="../output.csv")
    parser.add_argument("--dataset-dir", type=str, default="../dataset")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output
    dataset_dir = os.path.join(base_dir, args.dataset_dir) if not os.path.isabs(args.dataset_dir) else args.dataset_dir

    print(f"Loading static data from {dataset_dir}...")
    load_static_data(dataset_dir)

    print(f"Reading input from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Total rows read from claims.csv: {len(df)}")
    
    # No filter for final run, but just evaluating all rows with threads
    total_rows = len(df)

    output_rows = [None] * total_rows
    start_time = time.time()

    print(f"\n=== Processing {total_rows} rows via ThreadPool (max_workers=5) ===")

    def worker(idx_and_row):
        idx, row = idx_and_row
        time.sleep(2.0)  # Small per-worker delay to avoid bursting
        print(f"Processing row {idx + 1}/{total_rows} - user_id: {row['user_id']} ...")
        try:
            out_row = process_row(row, dataset_dir)
            return idx, out_row.model_dump(by_alias=True)
        except Exception as e:
            print(f"Error processing row {idx + 1}: {e}")
            fallback = OutputRow(
                user_id=row['user_id'],
                image_paths=row['image_paths'],
                user_claim=row['user_claim'],
                claim_object=row['claim_object'],
                evidence_standard_met="false",
                evidence_standard_met_reason="Unable to process claim due to an unexpected error.",
                risk_flags="manual_review_required",
                issue_type="unknown",
                object_part="unknown",
                claim_status="not_enough_information",
                claim_status_justification="This claim requires manual review due to a system error.",
                supporting_image_ids="none",
                valid_image="false",
                severity="unknown"
            )
            return idx, fallback.model_dump(by_alias=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(worker, item): item for item in df.iterrows()}
        for future in concurrent.futures.as_completed(futures):
            idx, res = future.result()
            output_rows[idx] = res

    out_df = pd.DataFrame(output_rows)
    columns_order = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids",
        "valid_image", "severity"
    ]
    out_df = out_df[columns_order]

    print(f"\nWriting output to {output_path}...")
    print(f"Total rows written to output.csv: {len(out_df)}")
    out_df.to_csv(output_path, index=False, mode="w", encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"\nDone! Processed {total_rows} rows in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
