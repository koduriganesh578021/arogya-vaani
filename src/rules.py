"""
Deterministic rule-based preliminary eligibility pre-check.

NO LLM involved. Pure Python rules. Output is INDICATIVE ONLY — never an
official eligibility decision.

Each rule returns one of:
  - "met"          : the factor is present
  - "not_met"      : the factor is clearly absent
  - "unclear"      : not enough information
"""

from typing import Literal

RuleStatus = Literal["met", "not_met", "unclear"]


# ---------------------------------------------------------------------
# PM-JAY rules
# ---------------------------------------------------------------------
def check_pmjay(
    *,
    has_aadhaar: bool,
    has_ration_card: bool,
    has_secc_listing: bool,
    is_bpl_family: bool,
    family_size: int,
) -> list[dict]:
    rules = []

    rules.append({
        "factor": "Aadhaar card",
        "status": "met" if has_aadhaar else "not_met",
        "reason": (
            "Aadhaar is the primary identity document for PM-JAY e-KYC."
            if has_aadhaar else
            "Aadhaar is required for PM-JAY e-KYC. If missing, apply at an "
            "Aadhaar enrolment centre first."
        ),
    })

    rules.append({
        "factor": "Ration card",
        "status": "met" if has_ration_card else "not_met",
        "reason": (
            "Ration card helps establish family identity and is commonly "
            "checked at the helpdesk."
            if has_ration_card else
            "Ration card is not strictly mandatory if SECC listing is "
            "present, but it speeds up verification. Apply at the local "
            "Civil Supplies office if missing."
        ),
    })

    rules.append({
        "factor": "SECC 2011 listing",
        "status": "met" if has_secc_listing else "unclear",
        "reason": (
            "Being on the SECC 2011 list (D1–D7 categories) is the primary "
            "PM-JAY eligibility basis."
            if has_secc_listing else
            "SECC listing cannot be verified in this session. Check at the "
            "nearest Common Service Centre or PM-JAY helpdesk."
        ),
    })

    rules.append({
        "factor": "BPL (Below Poverty Line) status",
        "status": "met" if is_bpl_family else "not_met",
        "reason": (
            "PM-JAY targets BPL and vulnerable families."
            if is_bpl_family else
            "PM-JAY is designed for BPL families. Some non-BPL families may "
            "still qualify via occupational categories — verify at the helpdesk."
        ),
    })

    if family_size < 1:
        rules.append({
            "factor": "Family size",
            "status": "unclear",
            "reason": "Family size was not provided.",
        })
    else:
        rules.append({
            "factor": "Family size",
            "status": "met",
            "reason": (
                f"Family size recorded as {family_size}. All members listed "
                f"on the household card are typically covered."
            ),
        })

    return rules


# ---------------------------------------------------------------------
# Aarogyasri rules
# ---------------------------------------------------------------------
def check_aarogyasri(
    *,
    has_aadhaar: bool,
    has_white_ration_card: bool,
    is_telangana_resident: bool,
) -> list[dict]:
    rules = []

    rules.append({
        "factor": "Telangana residency",
        "status": "met" if is_telangana_resident else "not_met",
        "reason": (
            "Aarogyasri is a Telangana state scheme for Telangana residents."
            if is_telangana_resident else
            "Aarogyasri is only available to Telangana residents."
        ),
    })

    rules.append({
        "factor": "White Ration Card",
        "status": "met" if has_white_ration_card else "not_met",
        "reason": (
            "The White Ration Card is the primary eligibility document for "
            "Aarogyasri."
            if has_white_ration_card else
            "The White Ration Card is the primary eligibility document. Apply "
            "at the local Civil Supplies office if missing."
        ),
    })

    rules.append({
        "factor": "Aadhaar card",
        "status": "met" if has_aadhaar else "not_met",
        "reason": (
            "Aadhaar is required for beneficiary identification at the "
            "Aarogyamithra desk."
            if has_aadhaar else
            "Aadhaar is required at the Aarogyamithra desk for beneficiary "
            "identification."
        ),
    })

    return rules


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def summarize(rules: list[dict]) -> dict:
    """Aggregate rule statuses into a preliminary summary."""
    met = sum(1 for r in rules if r["status"] == "met")
    not_met = sum(1 for r in rules if r["status"] == "not_met")
    unclear = sum(1 for r in rules if r["status"] == "unclear")
    total = len(rules)

    if not_met == 0 and unclear == 0:
        summary = "preliminary_positive"
        summary_text = (
            "Based on the information provided, the family appears to meet "
            "several preliminary indicators. Final eligibility must be "
            "confirmed at an authorized helpdesk."
        )
    elif not_met > 0:
        summary = "preliminary_missing"
        summary_text = (
            f"{not_met} of {total} key documents or factors are missing. "
            f"The family may need to arrange these before applying. Final "
            f"eligibility must be confirmed at an authorized helpdesk."
        )
    else:
        summary = "preliminary_unclear"
        summary_text = (
            "Some information could not be verified in this session. Please "
            "visit an authorized helpdesk for a complete assessment."
        )

    return {
        "summary": summary,
        "summary_text": summary_text,
        "counts": {
            "met": met,
            "not_met": not_met,
            "unclear": unclear,
            "total": total,
        },
    }
