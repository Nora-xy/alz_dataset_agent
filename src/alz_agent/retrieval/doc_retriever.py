"""P4 — Documentation evidence retriever.

Public interface: retrieve_doc_evidence(query, k, corpus_dir).
Returns list[DocEvidence] with DOC-kind evidence for grounding definitional claims.
Returns [] when no corpus is available, rank_bm25 is not installed, or no result
exceeds the relevance threshold — the orchestrator treats this as ungrounded.

No fabrication: if there is nothing to return, the list is empty.

Upgrade path: swap _load_corpus() for a PDF-based extractor without touching
the public interface or any P5 code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from alz_agent.schemas import DocEvidence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BM25_RELEVANCE_THRESHOLD: float = 0.1

# Default corpus directory: project_root/retrieval_notes/
# Tests override this via the corpus_dir parameter.
_DEFAULT_CORPUS_DIR: Path = (
    Path(__file__).parent.parent.parent.parent / "retrieval_notes"
)

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def retrieve_doc_evidence(
    query: str,
    k: int = 5,
    corpus_dir: Path | str | None = None,
) -> list[DocEvidence]:
    """Return up to k DocEvidence entries matching query.

    Args:
        query:      Natural-language query string.
        k:          Maximum number of results to return.
        corpus_dir: Directory containing .txt corpus files organised in
                    sub-folders (sub-folder name → source, file stem → locator).
                    Defaults to <project_root>/retrieval_notes/.

    Returns:
        List of DocEvidence sorted by descending relevance. Empty list when:
          - corpus_dir does not exist or is empty
          - rank_bm25 is not installed
          - no result exceeds BM25_RELEVANCE_THRESHOLD
    """
    resolved_dir = Path(corpus_dir) if corpus_dir is not None else _DEFAULT_CORPUS_DIR
    corpus = _load_corpus(resolved_dir)
    if not corpus:
        return []

    try:
        # BM25Plus uses log(1+IDF) which stays positive for small corpora.
        from rank_bm25 import BM25Plus  # optional dependency
    except ImportError:
        return []

    tokenized = [_tokenize(doc[2]) for doc in corpus]
    bm25 = BM25Plus(tokenized)
    scores = bm25.get_scores(_tokenize(query))

    results: list[DocEvidence] = []
    for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]:
        if scores[idx] < BM25_RELEVANCE_THRESHOLD:
            break
        source, locator, text = corpus[idx]
        results.append(DocEvidence(
            source=source,
            locator=locator,
            snippet=text[:300].strip(),
        ))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_corpus(corpus_dir: Path) -> list[tuple[str, str, str]]:
    """Load (source, locator, text) triples from corpus_dir.

    Walks corpus_dir recursively; reads every .txt file.
    source  = path of the file's parent relative to corpus_dir (or corpus_dir name
              if the file is directly in corpus_dir).
    locator = file stem (filename without extension).
    text    = full file content.

    Returns [] if corpus_dir does not exist or contains no .txt files.
    """
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        return []

    entries: list[tuple[str, str, str]] = []
    for txt_file in sorted(corpus_dir.rglob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        rel_parent = txt_file.parent.relative_to(corpus_dir)
        source = str(rel_parent) if str(rel_parent) != "." else corpus_dir.name
        locator = txt_file.stem
        entries.append((source, locator, text))
    return entries


def _tokenize(text: str) -> list[str]:
    return text.lower().split()
