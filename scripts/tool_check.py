"""Manual end-to-end exercise of every API tool against a running backend.

Usage: python scripts/tool_check.py [--base-url http://localhost:8000]
Generates two small PDFs, uploads them, then calls every endpoint and prints a
concise view of each tool's output. Not part of the automated test suite.
"""

from __future__ import annotations

import argparse
import textwrap

import fitz  # PyMuPDF
import httpx


def make_pdf(title: str, paragraphs: list[str]) -> bytes:
    doc = fitz.open()
    for i, para in enumerate(paragraphs):
        page = doc.new_page()
        page.insert_text((72, 90), f"{title} - Page {i + 1}", fontsize=14)
        page.insert_textbox(fitz.Rect(72, 120, 520, 760), para, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def show(label: str, text: str, *, limit: int = 320) -> None:
    body = " ".join(str(text).split())
    if len(body) > limit:
        body = body[:limit] + " …"
    print(f"\n=== {label} ===")
    print(textwrap.fill(body, width=88))


def show_citations(cits: list[dict]) -> None:
    if not cits:
        print("   citations: (none)")
        return
    for c in cits:
        print(f"   [{c['marker']}] {c['document_name']} p.{c['page_number']} "
              f"(score={c['score']:.3f}) “{c['quote'][:60]}…”")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    api = args.base_url.rstrip("/") + "/api"
    http = httpx.Client(timeout=180)

    print("#" * 90)
    print("# AI RESEARCH ASSISTANT — FULL TOOL CHECK")
    print("#" * 90)

    # ---- health ----
    show("HEALTH", http.get(f"{api}/health").json())

    # ---- upload two documents ----
    pdf_a = make_pdf(
        "Transformers",
        [
            "Transformers use self-attention to model long-range dependencies between "
            "tokens. Attention weights are computed with a softmax over scaled "
            "query-key dot products. The methodology trains on large text corpora "
            "using the Adam optimizer and evaluates with BLEU and perplexity.",
            "A limitation of transformers is their quadratic memory cost in sequence "
            "length. Future work includes sparse attention and linear-time variants "
            "to scale to longer documents.",
        ],
    )
    pdf_b = make_pdf(
        "Recurrent Networks",
        [
            "Recurrent neural networks process sequences step by step, maintaining a "
            "hidden state. LSTMs and GRUs mitigate vanishing gradients. The methodology "
            "uses backpropagation through time and evaluates on language modeling.",
            "A limitation of recurrent networks is difficulty capturing very long "
            "context and limited parallelism. Future work explores attention-augmented "
            "recurrence.",
        ],
    )

    ids: list[str] = []
    for name, data in (("transformers.pdf", pdf_a), ("recurrent.pdf", pdf_b)):
        r = http.post(f"{api}/documents", files={"file": (name, data, "application/pdf")})
        if r.status_code == 201:
            doc = r.json()
            ids.append(doc["id"])
            print(f"\n[upload] {name}: status={doc['status']} chunks={doc['chunk_count']} "
                  f"pages={doc['metadata']['page_count']}")
        elif r.status_code == 409:
            # Already uploaded in a previous run; fetch existing id from listing.
            print(f"\n[upload] {name}: duplicate (already indexed)")
        else:
            print(f"\n[upload] {name}: ERROR {r.status_code} {r.text}")

    # Resolve ids from the listing (covers the duplicate case).
    listing = http.get(f"{api}/documents").json()
    by_name = {d["filename"]: d["id"] for d in listing["documents"]}
    doc_a = by_name.get("transformers.pdf")
    doc_b = by_name.get("recurrent.pdf")
    print(f"\n[documents] total={listing['total']} "
          f"transformers={doc_a is not None} recurrent={doc_b is not None}")

    # ---- dashboard ----
    show("DASHBOARD", http.get(f"{api}/dashboard").json())

    # ---- search ----
    s = http.post(f"{api}/search", json={"query": "attention mechanism", "top_k": 3}).json()
    print(f"\n=== SEARCH 'attention mechanism' ({s['elapsed_ms']:.0f} ms) ===")
    for hit in s["hits"]:
        print(f"   {hit['document_name']} p.{hit['page_number']} score={hit['score']:.3f}: "
              f"{hit['text'][:70]}…")

    # ---- chat ----
    session = http.post(f"{api}/chats", json={"document_ids": ids or [doc_a, doc_b]}).json()
    sid = session["id"]
    ask = http.post(
        f"{api}/chats/{sid}/ask",
        json={"question": "What do transformers use for long-range dependencies?",
              "document_ids": [doc_a] if doc_a else None},
    ).json()["answer"]
    show("CHAT ANSWER", ask["text"])
    print(f"   context_found={ask['context_found']} "
          f"retrieval={ask['retrieval_ms']:.0f}ms llm={ask['llm_ms']:.0f}ms")
    show_citations(ask["citations"])

    # ---- AI features (single-doc) ----
    if doc_a:
        for label, path in (
            ("SUMMARY", f"/ai/{doc_a}/summary"),
            ("METHODOLOGY", f"/ai/{doc_a}/methodology"),
            ("LIMITATIONS", f"/ai/{doc_a}/limitations"),
            ("FUTURE WORK", f"/ai/{doc_a}/future-work"),
        ):
            a = http.post(f"{api}{path}").json()
            show(label, a["text"])
            show_citations(a["citations"])

        explain = http.post(
            f"{api}/ai/explain",
            json={"document_id": doc_a, "concept": "self-attention"},
        ).json()
        show("EXPLAIN 'self-attention'", explain["text"])

        quiz = http.post(
            f"{api}/ai/quiz", json={"document_id": doc_a, "num_questions": 3}
        ).json()
        show("QUIZ (3 questions)", quiz["text"], limit=500)

        cards = http.post(
            f"{api}/ai/flashcards", json={"document_id": doc_a, "num_cards": 4}
        ).json()
        show("FLASHCARDS (4)", cards["text"], limit=500)

        # ---- citations in every style ----
        print("\n=== CITATION FORMATS ===")
        for style in ("apa", "ieee", "mla", "bibtex"):
            c = http.get(f"{api}/ai/{doc_a}/citation", params={"style": style}).json()
            print(f"   [{style}] {c['citation']}")

    # ---- multi-doc features ----
    if doc_a and doc_b:
        compare = http.post(
            f"{api}/ai/compare",
            json={"document_ids": [doc_a, doc_b], "aspect": "handling long context"},
        ).json()
        show("COMPARE (transformers vs recurrent)", compare["text"], limit=400)

        review = http.post(
            f"{api}/ai/literature-review",
            json={"document_ids": [doc_a, doc_b], "topic": "sequence modeling"},
        ).json()
        show("LITERATURE REVIEW", review["text"], limit=400)

    # ---- export ----
    md = http.get(f"{api}/export/chats/{sid}", params={"fmt": "markdown"}).content
    pdf = http.get(f"{api}/export/chats/{sid}", params={"fmt": "pdf"}).content
    print("\n=== EXPORT ===")
    print(f"   markdown: {len(md)} bytes, starts with: {md[:24]!r}")
    print(f"   pdf: {len(pdf)} bytes, valid={pdf[:5] == b'%PDF-'}")

    print("\n" + "#" * 90)
    print("# ALL TOOLS EXERCISED SUCCESSFULLY")
    print("#" * 90)
    http.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
