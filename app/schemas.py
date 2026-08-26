from pydantic import BaseModel, Field, model_validator


class AnalyzeRequest(BaseModel):
    text: str | None = Field(None, description="Article text or claim to analyze")
    url: str | None = Field(None, description="URL of a news article to fetch and analyze")

    @model_validator(mode="after")
    def check_input_provided(self):
        if not self.text and not self.url:
            raise ValueError("Provide either 'text' or 'url'")
        if self.text and len(self.text.strip()) < 10:
            raise ValueError("text must be at least 10 characters")
        return self


class ClaimEvidence(BaseModel):
    claim: str
    fact_checks: list[dict]


class AnalyzeResponse(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    claims: list[str]
    evidence: list[ClaimEvidence]
    sources: list[str]
    style_flags: list[str]
    style_score: float
    dataset_match: bool = False
    dataset_label: str | None = None
    dataset_similarity: float = 0.0


class HistoryEntry(BaseModel):
    id: int
    created_at: str
    input_preview: str
    verdict: str
    confidence: float
    result: AnalyzeResponse


class QuestionRequest(BaseModel):
    question: str = Field(..., description="A question to answer from PDFs, then the web")

    @model_validator(mode="after")
    def check_question_length(self):
        if len(self.question.strip()) < 5:
            raise ValueError("question must be at least 5 characters")
        return self


class QAResponse(BaseModel):
    answer: str
    source: str = Field(..., description="'pdf', 'web', or 'none'")
    citations: list[str]
    confidence: float


class PDFUploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
