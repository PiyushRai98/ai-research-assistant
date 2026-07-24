"""Shared pytest fixtures.

Tests run entirely offline and deterministically by wiring the container with
the lightweight stack: :class:`HashingEmbedding`, :class:`NumpyVectorStore`,
and :class:`EchoClient`. This keeps the suite fast (no model downloads, no
network) while exercising the real application and infrastructure code paths.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from app.application.ai_features import AIFeatureService
from app.application.chat import ChatService
from app.application.documents import DocumentService
from app.application.rag import RAGService
from app.application.retrieval import RetrievalService
from app.backend.container import Container
from app.infrastructure.chat_store import SQLiteChatRepository
from app.infrastructure.chunking import RecursiveChunker
from app.infrastructure.database import SQLiteDocumentRepository
from app.infrastructure.embeddings import HashingEmbedding
from app.infrastructure.llm import EchoClient
from app.infrastructure.pdf import PyMuPDFParser
from app.infrastructure.vectorstore import NumpyVectorStore
from app.shared.config import LLMProvider, RetrievalStrategy, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings pointing all storage at a temp directory."""
    return Settings(
        app_env="development",
        storage_upload_dir=tmp_path / "uploads",
        storage_vectorstore_dir=tmp_path / "vectors",
        storage_database_path=tmp_path / "db" / "meta.db",
        storage_log_dir=tmp_path / "logs",
        chunk_size=400,
        chunk_overlap=60,
        retrieval_strategy=RetrievalStrategy.SIMILARITY,
        retrieval_top_k=4,
        llm_provider=LLMProvider.ECHO,
    )


@pytest.fixture
def container(settings: Settings) -> Container:
    """A fully wired, offline container for fast deterministic tests."""
    settings.ensure_directories()
    embedding = HashingEmbedding(dimension=256)
    store = NumpyVectorStore(settings.storage_vectorstore_dir)
    doc_repo = SQLiteDocumentRepository(settings.storage_database_path)
    chat_repo = SQLiteChatRepository(settings.storage_database_path)
    parser = PyMuPDFParser()
    chunker = RecursiveChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    llm = EchoClient()

    document_service = DocumentService(
        repository=doc_repo,
        parser=parser,
        chunker=chunker,
        embedding_model=embedding,
        vector_store=store,
        settings=settings,
    )
    retrieval_service = RetrievalService(
        embedding_model=embedding, vector_store=store, settings=settings
    )
    rag_service = RAGService(retrieval_service=retrieval_service, llm_client=llm)
    chat_service = ChatService(repository=chat_repo, rag_service=rag_service)
    ai_feature_service = AIFeatureService(rag_service=rag_service)

    return Container(
        settings=settings,
        embedding_model=embedding,
        vector_store=store,
        document_service=document_service,
        retrieval_service=retrieval_service,
        rag_service=rag_service,
        chat_service=chat_service,
        ai_feature_service=ai_feature_service,
    )


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a small, valid two-page PDF with known text via PyMuPDF."""
    import fitz

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Transformers use self-attention to model long-range dependencies. "
        "Attention weights are computed using a softmax over query-key scores.",
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Recurrent neural networks process sequences step by step and can "
        "struggle to retain information over long contexts.",
    )
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(container: Container, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """FastAPI TestClient with the offline container injected everywhere.

    ``get_container`` is patched (in both the container module and the main
    module that imported it) so ``create_app`` and its lifespan never build the
    heavyweight real container — the suite stays offline and fast.
    """
    import app.backend.container as container_module
    import app.backend.main as main_module
    from app.backend.dependencies import container_dep
    from fastapi.testclient import TestClient

    monkeypatch.setattr(container_module, "get_container", lambda: container)
    monkeypatch.setattr(main_module, "get_container", lambda: container)

    app = main_module.create_app()
    app.dependency_overrides[container_dep] = lambda: container
    with TestClient(app) as test_client:
        yield test_client
