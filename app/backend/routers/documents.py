"""Document upload, listing, retrieval, and deletion endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse

from app.backend.dependencies import DocumentServiceDep, OwnerDep
from app.backend.schemas import (
    DocumentListResponse,
    DocumentResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a PDF",
)
async def upload_document(
    service: DocumentServiceDep,
    owner: OwnerDep,
    file: Annotated[UploadFile, File(...)],
) -> DocumentResponse:
    """Upload a PDF; it is validated, deduplicated, chunked, embedded, indexed."""
    data = await file.read()
    document = service.upload(
        data=data,
        filename=file.filename or "document.pdf",
        content_type=file.content_type,
        owner=owner,
    )
    return DocumentResponse.from_domain(document)


@router.get("", response_model=DocumentListResponse, summary="List documents")
def list_documents(service: DocumentServiceDep, owner: OwnerDep) -> DocumentListResponse:
    documents = service.list(owner=owner)
    return DocumentListResponse(
        documents=[DocumentResponse.from_domain(d) for d in documents],
        total=len(documents),
    )


@router.get("/{document_id}", response_model=DocumentResponse, summary="Get a document")
def get_document(
    document_id: str, service: DocumentServiceDep, owner: OwnerDep  # noqa: ARG001
) -> DocumentResponse:
    return DocumentResponse.from_domain(service.get(document_id))


@router.get("/{document_id}/file", summary="Download the original PDF")
def get_document_file(
    document_id: str, service: DocumentServiceDep, owner: OwnerDep  # noqa: ARG001
) -> FileResponse:
    document = service.get(document_id)
    path = service.file_path(document_id)
    return FileResponse(path, media_type="application/pdf", filename=document.filename)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete a document",
)
def delete_document(
    document_id: str, service: DocumentServiceDep, owner: OwnerDep  # noqa: ARG001
) -> None:
    service.delete(document_id)
