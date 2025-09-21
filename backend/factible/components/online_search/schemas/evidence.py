from typing import List, Literal, Optional

from pydantic import BaseModel, Field


EvidenceStance = Literal["supports", "refutes", "mixed", "unclear"]


class EvidenceSnippet(BaseModel):
    """Relevant fragment extracted from a source document."""

    text: str = Field(description="Verbatim snippet from the source")
    rationale: Optional[str] = Field(
        default=None, description="Why this snippet matters for the query"
    )
    stance: EvidenceStance = Field(
        default="unclear",
        description="Whether this snippet supports, refutes, is mixed on, or is unclear about the claim.",
    )


class EvidenceExtraction(BaseModel):
    """Structured output produced by the evidence extractor."""

    has_relevant_evidence: bool = Field(description="True when content matches query")
    summary: Optional[str] = Field(
        default=None, description="Brief synthesis of evidence"
    )
    snippets: List[EvidenceSnippet] = Field(default_factory=list)
    overall_stance: EvidenceStance = Field(
        default="unclear",
        description="Overall stance of the gathered evidence towards the claim",
    )
