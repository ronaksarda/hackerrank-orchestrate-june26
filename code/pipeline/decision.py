"""
decision.py — Static data loading and helper functions.
Loads user_history.csv and evidence_requirements.csv once,
provides lookup functions used by the orchestrator and analyzer.
"""
import pandas as pd
import os
from typing import List

_USER_HISTORY = None
_EVIDENCE_REQ = None


def load_static_data(dataset_dir: str):
    """Load user_history.csv and evidence_requirements.csv into memory."""
    global _USER_HISTORY, _EVIDENCE_REQ
    if _USER_HISTORY is None:
        path = os.path.join(dataset_dir, "user_history.csv")
        _USER_HISTORY = pd.read_csv(path).set_index("user_id").to_dict(orient="index")
    if _EVIDENCE_REQ is None:
        path = os.path.join(dataset_dir, "evidence_requirements.csv")
        _EVIDENCE_REQ = pd.read_csv(path).to_dict(orient="records")


def get_evidence_req_desc(claim_object: str, expected_issue: str = "") -> str:
    """Build a human-readable string of all applicable evidence requirements."""
    if not _EVIDENCE_REQ:
        return ""
    reqs = [r['minimum_image_evidence'] for r in _EVIDENCE_REQ if r['claim_object'] in ['all', claim_object]]
    return " ".join(reqs)


def get_user_history_context(user_id: str) -> dict:
    """Return the full user history row as a dict, or empty dict if not found."""
    if _USER_HISTORY and user_id in _USER_HISTORY:
        return _USER_HISTORY[user_id]
    return {}


def get_user_risk_flags(user_id: str) -> List[str]:
    """
    Derive risk flags from user history.
    user_history_risk is added ONLY when rejected_claim > 0.
    """
    flags = []
    user_data = get_user_history_context(user_id)
    if not user_data:
        return flags

    # Explicit flags from history_flags column
    flags_str = user_data.get("history_flags", "none")
    if pd.notna(flags_str) and str(flags_str).strip() != "none":
        for f in str(flags_str).split(";"):
            f = f.strip()
            if f:
                flags.append(f)

    # user_history_risk: triggered ONLY by past rejections
    rejected = user_data.get("rejected_claim", 0)
    if rejected > 0:
        flags.append("user_history_risk")

    return list(set(flags))
