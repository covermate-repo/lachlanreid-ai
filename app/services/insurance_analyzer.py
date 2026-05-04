from typing import List, Literal, Optional
import re
from openai import OpenAI
from pydantic import BaseModel
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_CONTENT_CHARS = 80_000   # ~20k tokens — safe ceiling for gpt-4o-mini
EXPLANATION_MIN = 3
EXPLANATION_MAX = 6
POLICY_NOTES_MIN = 6
POLICY_NOTES_MAX = 12


# ── Structured output schema ───────────────────────────────────────────────────
class CoverageAnalysisOutput(BaseModel):
    policy_name: str
    user_question: str
    direct_answer: Literal[
        "highly unlikely",
        "unlikely",
        "likely",
        "very likely",
    ]
    explanation: List[str]
    explanation_summary: str
    policy_notes: List[str]
    policy_price: str
    final_summary: str


# ── Main service ───────────────────────────────────────────────────────────────
class InsuranceAnalyzer:
    """Service for analysing insurance coverage based on policy documents."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)

    # ── Text helpers ───────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """
        Lightly clean extracted PDF text without corrupting structure.
        Avoids wrapping mid-sentence words in === headers (previous bug).
        """
        # Collapse excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        # Strip trailing whitespace per line
        text = '\n'.join(line.rstrip() for line in text.splitlines())
        return text.strip()

    def truncate_content(self, text: str) -> str:
        """
        Hard-cap content to MAX_CONTENT_CHARS to avoid context overflow.
        Logs a warning if truncation occurs.
        """
        if len(text) > MAX_CONTENT_CHARS:
            logger.warning(
                "Content truncated from %d to %d chars to stay within context limit",
                len(text), MAX_CONTENT_CHARS
            )
            return text[:MAX_CONTENT_CHARS] + "\n\n[NOTE: Document truncated due to length]"
        return text

    def _normalize_list(self, value) -> List[str]:
        """Normalise model output into a clean list of non-empty single-line strings."""
        if isinstance(value, str):
            items = [line.strip(" -•\t") for line in value.splitlines() if line.strip()]
        elif isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            return []

        cleaned = []
        for item in items:
            single_line = re.sub(r"\s+", " ", item).strip()
            if single_line:
                cleaned.append(single_line)
        return cleaned

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate_response(self, result: dict, question: str, insurance_type: str) -> dict:
        """
        Validate and repair model output.
        Fixes: None values, empty fields, broken force-prefixes (removed).
        """
        policy_name = str(result.get("policy_name") or "N/A").strip() or "N/A"
        user_question = str(result.get("user_question") or question).strip() or question

        # Direct answer — likelihood scale
        direct_answer = str(result.get("direct_answer") or "").strip().lower()
        valid_likelihoods = {"highly unlikely", "unlikely", "likely", "very likely"}
        if direct_answer not in valid_likelihoods:
            direct_answer = "unable to determine"

        # Explanation points
        explanation = self._normalize_list(result.get("explanation"))
        explanation = explanation[:EXPLANATION_MAX]
        while len(explanation) < EXPLANATION_MIN:
            explanation.append(
                "Coverage outcome depends on the documented cause of loss and applicable exclusions."
            )

        # Explanation summary — do NOT force-prefix; trust the model
        explanation_summary = str(result.get("explanation_summary") or "").strip()
        if not explanation_summary:
            explanation_summary = (
                "Coverage depends on the proven cause of loss under the relevant "
                "insured-event and exclusion clauses."
            )

        # Policy notes
        policy_notes = self._normalize_list(result.get("policy_notes"))
        policy_notes = policy_notes[:POLICY_NOTES_MAX]

        # Policy price — from document only
        raw_price = result.get("policy_price")
        if not raw_price or str(raw_price).strip().lower() in ("none", "n/a", "null", "", "not available"):
            policy_price = "Not listed in provided documents"
        else:
            policy_price = str(raw_price).strip()

        # Final summary — do NOT force-prefix; trust the model
        final_summary = str(result.get("final_summary") or "").strip()
        if not final_summary:
            final_summary = (
                "Coverage is determined by the policy's insuring clauses, "
                "exclusions, and conditions."
            )

        return {
            "policy_name": policy_name,
            "user_question": user_question,
            "direct_answer": direct_answer,
            "explanation": explanation,
            "explanation_summary": explanation_summary,
            "policy_notes": policy_notes,
            "policy_price": policy_price,
            "final_summary": final_summary,
        }

    # ── Prompt ─────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return """You are an insurance policy analysis assistant for Covermate.

ROLE:
- Analyse policy documents and return a structured, factual report based ONLY on provided policy data.
- Do NOT provide advice, recommendations, or opinions.
- Do NOT hallucinate clause names — use only clause names present in the document.
- Do NOT infer beyond the wording present in the documents.
- Use plain, direct language.

STYLE RULES:
- Never use: "you should", "we recommend", "consider", or any advisory language.
- Clearly separate what triggers cover from what excludes it.
- Be concise and non-repetitive.
- If the policy is silent on the question, state that explicitly."""

    def _build_user_prompt(self, structured_content: str, question: str, insurance_type: str) -> str:
        return f"""INSURANCE DOCUMENTS:
{structured_content}

USER QUESTION: {question}

INSURANCE TYPE: {insurance_type.title()} Insurance

Return a JSON object matching this exact schema:

{{
    "policy_name": "Extract exact policy name from document. Use N/A if not found.",
    "user_question": "Repeat the user question verbatim.",
    "direct_answer": "Likelihood scale: 'highly unlikely' | 'unlikely' | 'likely' | 'very likely'. Based on how clearly the policy wording supports coverage for this scenario.",
    "explanation": [
        "3 to 6 bullet points. Reference specific clause names from the document only.",
        "Each point explains what triggers cover OR what excludes it.",
        "No advice, no speculation, no invented clauses."
    ],
    "explanation_summary": "One sentence: what the coverage outcome depends on.",
    "policy_notes": [
        "6 to 12 one-line notes describing the policy's GENERAL exclusions, limitations, sub-limits, and claim conditions.",
        "These should NOT be specific to the user question; they are broad policy warnings every holder should know.",
        "Format each as: <Issue> — <what it means> (<Clause Name>)",
        "Flag: exclusions, limits, claim constraints, structural gaps, things that are NOT covered regardless of cause."
    ],
    "policy_price": "If a price is visible in the documents, state it exactly. Otherwise: Not listed in provided documents",
    "final_summary": "One sentence summarising coverage determination by key clauses."
}}

IMPORTANT: Respond with valid JSON only. No markdown fences, no preamble."""

    # ── Core analysis ──────────────────────────────────────────────────────────

    def analyze_coverage(self, pdf_content: str, question: str, insurance_type: str) -> dict:
        """Analyse insurance coverage based on PDF content and user question."""

        logger.info("Analyzer: cleaning text (%d chars)", len(pdf_content))
        cleaned = self.clean_text(pdf_content)
        content = self.truncate_content(cleaned)
        logger.info("Analyzer: text ready (%d chars) — calling %s", len(content), self.settings.OPENAI_MODEL)

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user",   "content": self._build_user_prompt(content, question, insurance_type)},
                ],
                response_format=CoverageAnalysisOutput,
                temperature=0,          # deterministic output for factual analysis
                max_tokens=2048,
            )
            logger.info("Analyzer: OpenAI response received")

            parsed = response.choices[0].message.parsed
            if parsed is None:
                logger.error("Analyzer: structured output is None (refusal or parse failure)")
                return self._fallback_parse_failure(question, insurance_type)

            logger.info("Analyzer: structured output parsed successfully")
            validated = self.validate_response(parsed.model_dump(), question, insurance_type)
            logger.info(
                "Analyzer: validation complete — explanation=%d notes=%d",
                len(validated["explanation"]),
                len(validated["policy_notes"]),
            )
            return validated

        except Exception as e:
            logger.error("Analyzer: unexpected error — %s", str(e), exc_info=True)
            return self._fallback_technical_error(question, insurance_type)

    # ── Fallbacks ──────────────────────────────────────────────────────────────

    def _fallback_parse_failure(self, question: str, insurance_type: str) -> dict:
        """Returned when the model output could not be parsed."""
        return {
            "policy_name": "N/A",
            "user_question": question,
            "direct_answer": "unable to determine",
            "explanation": [
                "The model returned a response that could not be mapped to the required output structure.",
                "Coverage triggers and exclusions could not be extracted from this output.",
                "Manual review of the policy wording is required.",
            ],
            "explanation_summary": "Coverage depends on successful extraction of clause-level policy wording.",
            "policy_notes": [],
            "policy_price": "Not listed in provided documents",
            "final_summary": "Coverage is determined by the policy's insuring clauses, exclusions, and conditions.",
        }

    def _fallback_technical_error(self, question: str, insurance_type: str) -> dict:
        """Returned on API or network errors."""
        return {
            "policy_name": "N/A",
            "user_question": question,
            "direct_answer": "unable to determine",
            "explanation": [
                "A technical error occurred and the analysis could not be completed.",
                "Coverage trigger and exclusion mapping requires manual policy review.",
                "Please retry or contact support if the issue persists.",
            ],
            "explanation_summary": "Coverage depends on clause-level analysis that could not be completed.",
            "policy_notes": [],
            "policy_price": "Not listed in provided documents",
            "final_summary": "Coverage is determined by the policy's insuring clauses, exclusions, and conditions.",
        }