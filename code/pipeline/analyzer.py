"""
Single-shot comprehensive claim analyzer.
Makes ONE LLM call per row with full context (images + claim + history + requirements)
to produce the complete verdict.
"""
import json
import os
import base64
import hashlib
from openai import AzureOpenAI
from pydantic import BaseModel
from typing import List
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pipeline.cache_utils import load_cache, save_cache
from pipeline.image_utils import encode_image

ANALYZER_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", ".cache", "analyzer_cache.json")
_ANALYZER_CACHE = load_cache(ANALYZER_CACHE_PATH)

client = AzureOpenAI(
    azure_endpoint=os.getenv("OPENAI_BASE_URL", "https://foundry-misc-dev.services.ai.azure.com/").replace("/openai/v1", ""),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01"
)

def build_system_prompt(claim_object: str, evidence_req_desc: str, user_history_summary: str, has_user_risk: bool) -> str:
    history_note = ""
    if user_history_summary:
        history_note = f"\nUSER HISTORY: {user_history_summary}"
    if has_user_risk:
        history_note += "\nWARNING: This user has previously rejected claims. Add 'user_history_risk' to risk_flags."
    
    return f"""You are an expert insurance claims adjuster reviewing photographic evidence for a damage claim on a '{claim_object}'.
{history_note}

EVIDENCE REQUIREMENT: {evidence_req_desc}

Analyze the submitted image(s) alongside the claim conversation. Produce a JSON object with ALL of these keys:

1. "evidence_standard_met": "true" if the image set is sufficient to evaluate the claim, or if the image clearly CONTRADICTS the claim. Use "false" ONLY if the image is too blurry, dark, missing, or cropped to make ANY judgement.

2. "evidence_standard_met_reason": Short reason (1 sentence) for the evidence decision.

4. "risk_flags": Semicolon-separated list of applicable flags, or "none". Use ONLY:
   "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch", "possible_manipulation", "non_original_image", "text_instruction_present", "user_history_risk", "manual_review_required"
   - ALWAYS add "manual_review_required" if you use ANY of these: user_history_risk, claim_mismatch, non_original_image, wrong_object, text_instruction_present, possible_manipulation, damage_not_visible, cropped_or_obstructed, or if valid_image is false.
   - If different images appear to show different objects, vehicles, or devices that may not match each other (e.g. different car color, model, or context between images), include 'wrong_object' or 'claim_mismatch' in risk_flags and explain this in the justification, even if one image alone supports the claim.

5. "issue_type": The actual visible issue. Use ONLY: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown
   - Try to match the claimant's issue type if it's reasonably close (e.g., if they claim "scratch", use "scratch" even if it looks like a "dent"; if they claim "stain", use "stain" instead of "water_damage").
   - "none" = the claimed part IS visible and shows NO damage.
   - For side_mirror or headlight/taillight damage, prefer 'broken_part' over 'crack' unless the claim specifically concerns shattered glass/lens material rather than the housing or mounting. Reserve 'crack' primarily for glass surfaces: windshield, laptop screen, or similar flat glass/lens panels.
   - If the image contains circles, arrows, highlights, or other annotation marks pointing at an area, do NOT treat the mark itself as evidence of damage. Independently verify whether actual physical damage is visible in that area — the annotation only indicates where to look, not what is there.
   - For claims about missing contents or missing items inside a package: a partial view showing only packing material, with no full or unobstructed view of where the item would be, is NOT sufficient to confirm an item is missing. Use 'not_enough_information' unless the image clearly and comprehensively shows the entire interior is empty.

5. "object_part": The relevant part visible in the image. Use ONLY these values:
   Car: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, unknown
   Laptop: screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown
   Package: box, package_corner, package_side, seal, label, contents, item, unknown

6. "claim_status": Your final verdict. Use ONLY: supported, contradicted, not_enough_information
   - "supported" = image shows damage consistent with the claim.
   - "contradicted" = image clearly shows the claimed area with NO damage, or fundamentally conflicts (wrong object, wrong part).
   - "not_enough_information" = ONLY when the image is too blurry, too dark, or missing completely. If you can see clearly enough to know the claim is wrong, use "contradicted".
   - Compare the SEVERITY implied by the claimant's own description (e.g. 'looks pretty bad', 'badly damaged') against what is actually visible. If the conversation describes significant/severe damage but the image shows only minor or no damage on the correct part, this is a 'contradicted' claim due to severity mismatch — even if the part and general issue type are correct. Add 'claim_mismatch' to risk_flags in this case.
   - Authenticity concerns (non_original_image, possible_manipulation) do NOT automatically mean not_enough_information. If you can still clearly tell that the visible content does not match or support the specific claim — even in a stock photo, screenshot, or suspicious image — use 'contradicted', not 'not_enough_information'. Reserve not_enough_information for cases where the relevant area is genuinely unclear, obstructed, too blurry, or not visible at all — not for cases where you can see clearly that something is wrong.

7. "claim_status_justification": Concise explanation (1-2 sentences).

8. "supporting_image_ids": Semicolon-separated image IDs (e.g. "img_1") that support your decision. Use "none" if no image is sufficient.

9. "valid_image": "true" if the image is a genuine, usable photograph. "false" if it's a screenshot, digitally generated, completely irrelevant, or not a photo.

11. "severity": Rate ONLY the visible damage. Use ONLY: none, low, medium, high, unknown
    - "none" = no damage
    - "low" = minor cosmetic damage that does not affect function (small scratches, light surface marks, minor stains)
    - "medium" = moderate but contained damage (visible cracks, noticeable dents, torn packaging, clearly affected components)
    - "high" = severe or structural damage (shattered glass, completely broken/detached parts, heavy crushing, large missing sections)
    Judge severity from what is visible in the image, not from the issue_type label alone.
    A dent alone, even if clearly visible, is typically 'medium' severity unless the panel is torn, punctured, or structurally displaced — reserve 'high' for structural or non-cosmetic damage.

CRITICAL RULES:
- Output ONLY valid JSON, nothing else."""

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(5))
def _call_llm(system_prompt: str, user_content: list) -> dict:
    response = client.chat.completions.create(
        model='gpt-4.1',
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

def analyze_claim(
    user_id: str,
    image_paths_raw: str,
    user_claim: str,
    claim_object: str,
    dataset_dir: str,
    evidence_req_desc: str,
    user_history: dict
) -> dict:
    cache_key = hashlib.md5(f"V2_{user_id}_{image_paths_raw}_{claim_object}_{user_claim}".encode()).hexdigest()
    if cache_key in _ANALYZER_CACHE:
        return _ANALYZER_CACHE[cache_key]
    
    rejected = user_history.get("rejected_claim", 0)
    has_user_risk = rejected > 0
    history_summary = str(user_history.get("history_summary", "")) if user_history else ""
    
    system_prompt = build_system_prompt(claim_object, evidence_req_desc, history_summary, has_user_risk)
    
    image_paths = [p.strip() for p in image_paths_raw.split(';') if p.strip()]
    image_ids = []
    
    user_content = [
        {"type": "text", "text": f"CLAIM CONVERSATION:\n{user_claim}\n\nIMAGE COUNT: {len(image_paths)}"}
    ]
    
    img_detail = "low" if len(image_paths) > 1 else "high"
    
    for img_path in image_paths:
        full_path = os.path.join(dataset_dir, img_path)
        if not os.path.exists(full_path):
            continue
        
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        image_ids.append(img_id)
        
        b64 = encode_image(full_path)
        user_content.append({"type": "text", "text": f"Image ID: {img_id}"})
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": img_detail
            }
        })
    
    print(f"DEBUG row={user_id} images={image_paths} payload_size_kb={sum(len(c.get('image_url',{}).get('url','')) for c in user_content if c.get('type')=='image_url')/1024:.1f}")
    
    try:
        result = _call_llm(system_prompt, user_content)
    except Exception as e:
        if str(e).startswith("Unreadable image file"):
            result = {
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": str(e),
                "risk_flags": "non_original_image;manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "The image file is genuinely corrupted and unreadable.",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            }
            return result
            
        import traceback
        from tenacity import RetryError
        if isinstance(e, RetryError):
            real_e = e.last_attempt.exception()
            print(f"DEBUG EXCEPTION for {user_id}: {type(real_e).__name__} - {str(real_e)}")
            if hasattr(real_e, 'response') and hasattr(real_e.response, 'text'):
                print(f"DEBUG RESPONSE: {real_e.response.text}")
                
            error_str = str(real_e).lower()
            if "content_filter" in error_str or "jailbreak" in error_str or "responsible ai" in error_str:
                result = {
                    "evidence_standard_met": "true",
                    "evidence_standard_met_reason": "Claim text contained a flagged instruction-injection attempt; evaluating based on available context only.",
                    "risk_flags": "text_instruction_present;manual_review_required",
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "Automated visual analysis could not run because the claim text triggered a content safety filter; flagged for manual review.",
                    "supporting_image_ids": "none",
                    "valid_image": "false",
                    "severity": "unknown",
                }
                return result
        else:
            print(f"DEBUG EXCEPTION for {user_id}: {type(e).__name__} - {str(e)}")
            
        result = {
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": "Automated review failed due to a processing error; manual review required.",
            "risk_flags": "manual_review_required",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "The system could not complete automated image analysis for this claim.",
            "supporting_image_ids": "none",
            "valid_image": "false",
            "severity": "unknown",
        }
        return result

    raw_flags = str(result.get("risk_flags", "none")).strip()
    if raw_flags == "none":
        flags_set = set()
    else:
        flags_set = set(f.strip() for f in raw_flags.split(";") if f.strip())
    
    if has_user_risk:
        flags_set.add("user_history_risk")
    
    history_flags_str = str(user_history.get("history_flags", "none")) if user_history else "none"
    if history_flags_str and history_flags_str != "none" and history_flags_str != "nan":
        for f in history_flags_str.split(";"):
            f = f.strip()
            if f:
                flags_set.add(f)
    
    serious_flags = {"user_history_risk", "claim_mismatch", "non_original_image", 
                     "wrong_object", "text_instruction_present", "possible_manipulation",
                     "damage_not_visible", "cropped_or_obstructed"}
    valid_img = str(result.get("valid_image", "true")).lower()
    
    if flags_set & serious_flags or valid_img == "false":
        flags_set.add("manual_review_required")
    
    flags_set.discard("none")
    
    if not flags_set:
        result["risk_flags"] = "none"
    else:
        result["risk_flags"] = ";".join(sorted(flags_set))
    
    if not result.get("supporting_image_ids") or result["supporting_image_ids"] == "":
        result["supporting_image_ids"] = "none"
        
    _ANALYZER_CACHE[cache_key] = result
    save_cache(ANALYZER_CACHE_PATH, _ANALYZER_CACHE)

    return result
