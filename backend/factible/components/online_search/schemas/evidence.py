from typing import Literal, Optional

from pydantic import BaseModel, Field


EvidenceStance = Literal["supports", "refutes", "mixed", "unclear"]


class EvidenceExtraction(BaseModel):
    """Structured output produced by the evidence extractor."""

    has_relevant_evidence: bool = Field(
        description="True when content contains relevant evidence for the claim"
    )
    summary: Optional[str] = Field(
        default=None,
        description="1-2 sentence synthesis explaining how the evidence relates to the claim",
    )
    overall_stance: EvidenceStance = Field(
        default="unclear",
        description="Overall stance of the evidence towards the claim: supports, refutes, mixed, or unclear",
    )
    key_quote: Optional[str] = Field(
        default=None,
        description="Optional: one compelling verbatim quote from the source if available",
    )
