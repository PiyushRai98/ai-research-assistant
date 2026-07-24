"""AI research feature endpoints (summaries, quizzes, comparison, citations)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from app.application.ai_features import (
    SUPPORTED_CITATION_STYLES,
    format_citation,
)
from app.backend.dependencies import (
    AIFeatureServiceDep,
    DocumentServiceDep,
    OwnerDep,
)
from app.backend.schemas import (
    AnswerResponse,
    CitationFormatResponse,
    CompareRequest,
    ExplainRequest,
    FlashcardRequest,
    LiteratureReviewRequest,
    QuizRequest,
)

router = APIRouter(prefix="/ai", tags=["ai-features"])


@router.post("/{document_id}/summary", response_model=AnswerResponse)
def summarize(
    document_id: str, service: AIFeatureServiceDep, owner: OwnerDep  # noqa: ARG001
) -> AnswerResponse:
    return AnswerResponse.from_domain(service.summarize(document_id=document_id))


@router.post("/{document_id}/methodology", response_model=AnswerResponse)
def methodology(
    document_id: str, service: AIFeatureServiceDep, owner: OwnerDep  # noqa: ARG001
) -> AnswerResponse:
    return AnswerResponse.from_domain(service.extract_methodology(document_id=document_id))


@router.post("/{document_id}/limitations", response_model=AnswerResponse)
def limitations(
    document_id: str, service: AIFeatureServiceDep, owner: OwnerDep  # noqa: ARG001
) -> AnswerResponse:
    return AnswerResponse.from_domain(service.extract_limitations(document_id=document_id))


@router.post("/{document_id}/future-work", response_model=AnswerResponse)
def future_work(
    document_id: str, service: AIFeatureServiceDep, owner: OwnerDep  # noqa: ARG001
) -> AnswerResponse:
    return AnswerResponse.from_domain(service.extract_future_work(document_id=document_id))


@router.post("/explain", response_model=AnswerResponse)
def explain(
    request: ExplainRequest, service: AIFeatureServiceDep, owner: OwnerDep  # noqa: ARG001
) -> AnswerResponse:
    answer = service.explain_concept(concept=request.concept, document_id=request.document_id)
    return AnswerResponse.from_domain(answer)


@router.post("/quiz", response_model=AnswerResponse)
def quiz(
    request: QuizRequest, service: AIFeatureServiceDep, owner: OwnerDep  # noqa: ARG001
) -> AnswerResponse:
    answer = service.generate_quiz(
        document_id=request.document_id, num_questions=request.num_questions
    )
    return AnswerResponse.from_domain(answer)


@router.post("/flashcards", response_model=AnswerResponse)
def flashcards(
    request: FlashcardRequest, service: AIFeatureServiceDep, owner: OwnerDep  # noqa: ARG001
) -> AnswerResponse:
    answer = service.generate_flashcards(
        document_id=request.document_id, num_cards=request.num_cards
    )
    return AnswerResponse.from_domain(answer)


@router.post("/compare", response_model=AnswerResponse)
def compare(
    request: CompareRequest, service: AIFeatureServiceDep, owner: OwnerDep  # noqa: ARG001
) -> AnswerResponse:
    answer = service.compare_papers(document_ids=request.document_ids, aspect=request.aspect)
    return AnswerResponse.from_domain(answer)


@router.post("/literature-review", response_model=AnswerResponse)
def literature_review(
    request: LiteratureReviewRequest,
    service: AIFeatureServiceDep,
    owner: OwnerDep,  # noqa: ARG001
) -> AnswerResponse:
    answer = service.literature_review(document_ids=request.document_ids, topic=request.topic)
    return AnswerResponse.from_domain(answer)


@router.get("/{document_id}/citation", response_model=CitationFormatResponse)
def citation(
    document_id: str,
    documents: DocumentServiceDep,
    owner: OwnerDep,  # noqa: ARG001
    style: str = Query(default="apa", description="apa | ieee | mla | bibtex"),
) -> CitationFormatResponse:
    """Format a document's bibliographic citation in the requested style."""
    if style.lower() not in SUPPORTED_CITATION_STYLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported style. Choose one of {SUPPORTED_CITATION_STYLES}.",
        )
    document = documents.get(document_id)
    return CitationFormatResponse(
        document_id=document_id,
        style=style.lower(),
        citation=format_citation(document, style=style),
    )
