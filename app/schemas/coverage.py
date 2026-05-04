from pydantic import BaseModel
from typing import List


class CoverageRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Is storm damage to my roof covered under this policy?"
            }
        }


class CoverageResponse(BaseModel):
    policy_name: str = "N/A"
    user_question: str
    direct_answer: str          # Likelihood: highly unlikely | unlikely | likely | very likely
    explanation: List[str]
    explanation_summary: str
    policy_notes: List[str]
    policy_price: str
    final_summary: str

    class Config:
        json_schema_extra = {
            "example": {
                "policy_name": "CGU Steadfast Home Insurance (Listed Events Cover)",
                "user_question": "Am I covered for water coming through my walls?",
                "direct_answer": "unlikely",
                "explanation": [
                    "Storm, Flood, Rainwater, Wind covers water damage caused directly by storm or rainwater.",
                    "Storm, Flood, Rainwater, Wind excludes ingress caused by structural defects, faulty design, or workmanship.",
                    "Escape of Liquid covers sudden escape from pipes or fixtures but excludes gradual seepage.",
                    "Listed Events Cover means damage is only covered where linked to a named insured event.",
                ],
                "explanation_summary": (
                    "Coverage depends on whether water entry was caused by a listed insured event "
                    "or by a structural or gradual cause."
                ),
                "policy_notes": [
                    "Listed events limitation — only named events trigger cover; anything not listed is excluded (Listed Events Cover)",
                    "Gradual damage exclusion — seepage, rust, and wear-and-tear are not covered (Escape of Liquid / General Exclusions)",
                    "Structural defect exclusion — faulty design or workmanship losses are excluded (Storm, Flood, Rainwater, Wind)",
                    "Source repair exclusion — cost to repair the defective pipe or fitting is not covered (Escape of Liquid)",
                    "Unoccupied dwelling limit — cover reduced or void after 60 consecutive days unoccupied (Unoccupied Buildings)",
                    "Excess payable — a standard excess applies to every claim (Excess / Paying Claims)",
                ],
                "policy_price": "Not listed in provided documents",
                "final_summary": (
                    "Coverage is determined by the Storm, Flood, Rainwater, Wind and Escape of Liquid "
                    "clauses within a Listed Events Cover structure."
                ),
            }
        }


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model: str