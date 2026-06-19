from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"

class IssueType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"

class CarObjectPart(str, Enum):
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    DOOR = "door"
    HOOD = "hood"
    WINDSHIELD = "windshield"
    SIDE_MIRROR = "side_mirror"
    HEADLIGHT = "headlight"
    TAILLIGHT = "taillight"
    FENDER = "fender"
    QUARTER_PANEL = "quarter_panel"
    BODY = "body"
    UNKNOWN = "unknown"

class LaptopObjectPart(str, Enum):
    SCREEN = "screen"
    KEYBOARD = "keyboard"
    TRACKPAD = "trackpad"
    HINGE = "hinge"
    LID = "lid"
    CORNER = "corner"
    PORT = "port"
    BASE = "base"
    BODY = "body"
    UNKNOWN = "unknown"

class PackageObjectPart(str, Enum):
    BOX = "box"
    PACKAGE_CORNER = "package_corner"
    PACKAGE_SIDE = "package_side"
    SEAL = "seal"
    LABEL = "label"
    CONTENTS = "contents"
    ITEM = "item"
    UNKNOWN = "unknown"

class RiskFlag(str, Enum):
    NONE = "none"
    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    CLAIM_MISMATCH = "claim_mismatch"
    POSSIBLE_MANIPULATION = "possible_manipulation"
    NON_ORIGINAL_IMAGE = "non_original_image"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"

class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"

class OutputRow(BaseModel):
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str  # "true" or "false"
    evidence_standard_met_reason: str
    risk_flags: str  # semicolon separated or "none"
    issue_type: IssueType
    object_part: str
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: str  # semicolon separated or "none"
    valid_image: str  # "true" or "false"
    severity: Severity

    class Config:
        use_enum_values = True

# Helper to normalize object parts based on claim object
def normalize_object_part(part: str, claim_object: str) -> str:
    part_lower = part.lower().strip().replace(" ", "_")
    
    if claim_object == "car":
        try:
            return CarObjectPart(part_lower).value
        except ValueError:
            return CarObjectPart.UNKNOWN.value
    elif claim_object == "laptop":
        try:
            return LaptopObjectPart(part_lower).value
        except ValueError:
            return LaptopObjectPart.UNKNOWN.value
    elif claim_object == "package":
        try:
            return PackageObjectPart(part_lower).value
        except ValueError:
            return PackageObjectPart.UNKNOWN.value
    return "unknown"
