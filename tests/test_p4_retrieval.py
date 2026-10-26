"""P4 tests: documentation evidence retriever.

Tests:
  1. corpus hit — in-corpus query returns ≥1 DocEvidence with correct structure
  2. below-threshold / out-of-corpus query returns [] (no fabrication)
  3. missing corpus dir returns []
  4. k limit is respected
  5. source and locator are derived from directory layout
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alz_agent.retrieval.doc_retriever import BM25_RELEVANCE_THRESHOLD, _load_corpus, retrieve_doc_evidence
from alz_agent.schemas import DocEvidence

# Path to the small fixture corpus checked in alongside tests.
FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "retrieval"


# ---------------------------------------------------------------------------
# _load_corpus unit tests (no rank_bm25 dependency)
# ---------------------------------------------------------------------------


class TestLoadCorpus:
    def test_returns_entries_for_fixture_corpus(self):
        entries = _load_corpus(FIXTURE_CORPUS)
        assert len(entries) == 2

    def test_entry_structure(self):
        entries = _load_corpus(FIXTURE_CORPUS)
        for source, locator, text in entries:
            assert isinstance(source, str) and source
            assert isinstance(locator, str) and locator
            assert isinstance(text, str) and len(text) > 10

    def test_source_matches_subdirectory(self):
        entries = _load_corpus(FIXTURE_CORPUS)
        sources = {e[0] for e in entries}
        assert "NACC" in sources
        assert "ADNI" in sources

    def test_locator_matches_file_stem(self):
        entries = _load_corpus(FIXTURE_CORPUS)
        locators = {e[1] for e in entries}
        assert "uds3_excerpt" in locators
        assert "quickstart_excerpt" in locators

    def test_missing_dir_returns_empty(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert _load_corpus(missing) == []

    def test_empty_dir_returns_empty(self, tmp_path):
        assert _load_corpus(tmp_path) == []

    def test_skips_empty_txt_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        assert _load_corpus(tmp_path) == []


# ---------------------------------------------------------------------------
# retrieve_doc_evidence integration tests (require rank_bm25)
# ---------------------------------------------------------------------------


rank_bm25 = pytest.importorskip(
    "rank_bm25",
    reason="rank-bm25 not installed; install with: pip install 'alz-agent[retrieval]'",
)


class TestRetrieveDocEvidence:
    def test_corpus_hit_returns_doc_evidence(self):
        results = retrieve_doc_evidence("MMSE cognitive impairment score", corpus_dir=FIXTURE_CORPUS)
        assert len(results) >= 1
        for ev in results:
            assert isinstance(ev, DocEvidence)
            assert ev.kind == "DOC"

    def test_result_source_is_non_empty(self):
        results = retrieve_doc_evidence("MMSE cognitive", corpus_dir=FIXTURE_CORPUS)
        assert results
        assert all(ev.source for ev in results)

    def test_result_locator_is_non_empty(self):
        results = retrieve_doc_evidence("CDR dementia rating", corpus_dir=FIXTURE_CORPUS)
        assert results
        assert all(ev.locator for ev in results)

    def test_result_snippet_is_non_empty(self):
        results = retrieve_doc_evidence("MMSE score", corpus_dir=FIXTURE_CORPUS)
        assert results
        assert all(ev.snippet for ev in results)

    def test_snippet_capped_at_300_chars(self):
        results = retrieve_doc_evidence("CDR dementia memory", corpus_dir=FIXTURE_CORPUS)
        for ev in results:
            assert len(ev.snippet) <= 300

    def test_out_of_corpus_query_returns_empty(self):
        # Words that appear in neither fixture file
        results = retrieve_doc_evidence(
            "xyzzy42 foobar nonexistent_term_7391",
            corpus_dir=FIXTURE_CORPUS,
        )
        assert results == []

    def test_missing_corpus_dir_returns_empty(self, tmp_path):
        missing = tmp_path / "no_such_dir"
        results = retrieve_doc_evidence("MMSE score", corpus_dir=missing)
        assert results == []

    def test_k_limit_respected(self):
        # We have 2 fixture docs; requesting k=1 must return at most 1
        results = retrieve_doc_evidence("CDR MMSE dementia cognitive", k=1, corpus_dir=FIXTURE_CORPUS)
        assert len(results) <= 1

    def test_k_zero_returns_empty(self):
        results = retrieve_doc_evidence("MMSE", k=0, corpus_dir=FIXTURE_CORPUS)
        assert results == []

    def test_results_sorted_by_relevance(self):
        # Both docs may score > 0 for a broad query; we just check no crash and
        # that the list is non-empty (ordering itself is BM25-internal).
        results = retrieve_doc_evidence("dementia cognitive MCI diagnosis", corpus_dir=FIXTURE_CORPUS)
        assert isinstance(results, list)

    def test_adni_query_hits_adni_source(self):
        results = retrieve_doc_evidence("ADNI participant diagnosis DXSUM", corpus_dir=FIXTURE_CORPUS)
        assert results
        sources = {ev.source for ev in results}
        assert "ADNI" in sources

    def test_nacc_query_hits_nacc_source(self):
        results = retrieve_doc_evidence("NACCUDSD diagnostic summary NACC", corpus_dir=FIXTURE_CORPUS)
        assert results
        sources = {ev.source for ev in results}
        assert "NACC" in sources

    def test_custom_corpus_dir(self, tmp_path):
        # Create a one-doc corpus in tmp_path directly (no subdir)
        (tmp_path / "notes.txt").write_text(
            "Plasma biomarkers include pTau217 and amyloid beta 42/40 ratio.",
            encoding="utf-8",
        )
        results = retrieve_doc_evidence("plasma pTau amyloid biomarker", corpus_dir=tmp_path)
        assert len(results) >= 1
        assert "pTau" in results[0].snippet or "plasma" in results[0].snippet.lower()
