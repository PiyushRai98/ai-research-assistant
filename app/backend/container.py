"""Composition root: wires ports to concrete adapters exactly once.

This is the only module allowed to import both the application services and the
infrastructure adapters. It builds a single :class:`Container` holding fully
constructed, shared services, which FastAPI dependencies then hand out. Keeping
wiring here means the rest of the codebase depends only on abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.application.ai_features import AIFeatureService
from app.application.chat import ChatService
from app.application.documents import DocumentService
from app.application.rag import RAGService
from app.application.retrieval import RetrievalService
from app.domain.ports import EmbeddingModel, VectorStore
from app.infrastructure.chat_store import SQLiteChatRepository
from app.infrastructure.chunking import RecursiveChunker
from app.infrastructure.database import SQLiteDocumentRepository
from app.infrastructure.embeddings import build_embedding_model
from app.infrastructure.llm import build_llm_client
from app.infrastructure.pdf import PyMuPDFParser
from app.infrastructure.vectorstore import build_vector_store
from app.shared.config import Settings, get_settings
from app.shared.logging import configure_logging, get_logger

logger = get_logger("container")


@dataclass(slots=True)
class Container:
    """Holds the fully wired application services."""

    settings: Settings
    embedding_model: EmbeddingModel
    vector_store: VectorStore
    document_service: DocumentService
    retrieval_service: RetrievalService
    rag_service: RAGService
    chat_service: ChatService
    ai_feature_service: AIFeatureService


def build_container(settings: Settings | None = None) -> Container:
    """Construct all adapters and services from configuration."""
    settings = settings or get_settings()
    settings.ensure_directories()
    configure_logging(
        log_dir=settings.storage_log_dir,
        level=settings.app_log_level,
        json_logs=settings.is_production,
    )
    logger.info("Bootstrapping container for env={env}", env=settings.app_env.value)

    # --- infrastructure adapters ---
    embedding_model = build_embedding_model(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = build_vector_store(
        path=settings.storage_vectorstore_dir,
        dimension=embedding_model.dimension,
    )
    document_repo = SQLiteDocumentRepository(settings.storage_database_path)
    chat_repo = SQLiteChatRepository(settings.storage_database_path)
    parser = PyMuPDFParser()
    chunker = RecursiveChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    llm_client = build_llm_client(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )

    # --- application services ---
    document_service = DocumentService(
        repository=document_repo,
        parser=parser,
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
        settings=settings,
    )
    retrieval_service = RetrievalService(
        embedding_model=embedding_model,
        vector_store=vector_store,
        settings=settings,
    )
    rag_service = RAGService(retrieval_service=retrieval_service, llm_client=llm_client)
    chat_service = ChatService(repository=chat_repo, rag_service=rag_service)
    ai_feature_service = AIFeatureService(rag_service=rag_service)

    return Container(
        settings=settings,
        embedding_model=embedding_model,
        vector_store=vector_store,
        document_service=document_service,
        retrieval_service=retrieval_service,
        rag_service=rag_service,
        chat_service=chat_service,
        ai_feature_service=ai_feature_service,
    )


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Return the process-wide, lazily built container."""
    return build_container()
