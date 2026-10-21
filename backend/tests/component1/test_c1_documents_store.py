"""Tests for the document tables in component1/storage.py and for documents/candidates.py (SQLite)."""
import sqlite3

import pytest

from component1.documents.candidates import best_candidates, find_group, group_candidates
from component1.documents.chunk import Chunk
from component1.documents.extract import DocFact
from component1.storage import (
    create_document,
    delete_document,
    ensure_c1_tables,
    fail_document,
    finish_document,
    get_document,
    list_doc_chunks,
    list_doc_facts,
    list_documents,
    set_facts_status,
)


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(str(tmp_path / "c1.db"))
    ensure_c1_tables(c)
    yield c
    c.close()


def fact(field, value, page=1, chunk=1, span=None, hedged=False, method="rules"):
    return DocFact(chunk, page, field, value, span or value, hedged, method)


def upload(conn, name="Proposal.pdf", facts=(), team="t1"):
    doc = create_document(conn, team, name, 1234, uploaded_by="u1")
    finish_document(conn, doc["doc_id"], [Chunk(1, 1, "We will use XGBoost and RMSE.")], facts, pages=3)
    return doc["doc_id"]


# ------------------------------------------------------------------ storage
def test_an_upload_is_registered_as_processing_then_finished(conn):
    doc = create_document(conn, "t1", "Proposal.pdf", 1234, uploaded_by="u1")
    assert doc["status"] == "processing" and doc["filename"] == "Proposal.pdf"
    finish_document(conn, doc["doc_id"], [Chunk(1, 2, "chunk text here")], [fact("algorithm", "XGBoost", page=2)], pages=5)
    done = get_document(conn, "t1", doc["doc_id"])
    assert (done["status"], done["pages"], done["chunks"], done["facts"]) == ("done", 5, 1, 1)
    assert list_doc_chunks(conn, doc["doc_id"]) == [{"chunk_no": 1, "page": 2, "text": "chunk text here"}]
    (row,) = list_doc_facts(conn, "t1")
    assert (row["field"], row["value"], row["page"], row["filename"], row["status"], row["hedged"]) == \
        ("algorithm", "XGBoost", 2, "Proposal.pdf", "open", False)


def test_a_failed_document_keeps_its_reason(conn):
    doc = create_document(conn, "t1", "scan.pdf", 10)
    fail_document(conn, doc["doc_id"], "No text found in this PDF.")
    assert get_document(conn, "t1", doc["doc_id"])["status"] == "failed"
    assert get_document(conn, "t1", doc["doc_id"])["error"].startswith("No text")


def test_teams_only_see_their_own_documents(conn):
    upload(conn, team="t1")
    upload(conn, "Other.pdf", team="t2")
    assert [d["filename"] for d in list_documents(conn, "t1")] == ["Proposal.pdf"]
    assert get_document(conn, "t2", list_documents(conn, "t1")[0]["doc_id"]) is None


def test_deleting_keeps_accepted_facts_only(conn):
    doc_id = upload(conn, facts=[fact("algorithm", "XGBoost"), fact("evaluation_metric", "RMSE")])
    accepted = [f["fact_id"] for f in list_doc_facts(conn, "t1") if f["field"] == "algorithm"]
    set_facts_status(conn, "t1", accepted, "accepted", decided_by="u1")
    assert delete_document(conn, "t1", doc_id)
    assert list_documents(conn, "t1") == [] and list_doc_chunks(conn, doc_id) == []
    assert [f["field"] for f in list_doc_facts(conn, "t1")] == ["algorithm"]
    assert not delete_document(conn, "t1", doc_id)


def test_a_document_deleted_while_processing_stores_nothing(conn):
    doc = create_document(conn, "t1", "Proposal.pdf", 10)
    delete_document(conn, "t1", doc["doc_id"])
    finish_document(conn, doc["doc_id"], [Chunk(1, 1, "text text text")], [fact("algorithm", "XGBoost")], pages=1)
    assert list_doc_facts(conn, "t1") == [] and list_doc_chunks(conn, doc["doc_id"]) == []


def test_only_open_facts_change_status(conn):
    upload(conn, facts=[fact("algorithm", "XGBoost")])
    (f,) = list_doc_facts(conn, "t1")
    assert set_facts_status(conn, "t1", [f["fact_id"]], "dismissed", "u1") == 1
    assert set_facts_status(conn, "t1", [f["fact_id"]], "accepted", "u1") == 0
    assert list_doc_facts(conn, "t1")[0]["status"] == "dismissed"
    with pytest.raises(ValueError):
        set_facts_status(conn, "t1", [f["fact_id"]], "maybe")


# --------------------------------------------------------------- candidates
def test_the_most_mentioned_value_is_the_best_suggestion(conn):
    upload(conn, facts=[fact("algorithm", "Random Forest", page=1), fact("algorithm", "XGBoost", page=2),
                        fact("algorithm", "xgboost", page=4, chunk=2), fact("dataset", "UCI Retail", page=3)])
    best = best_candidates(list_doc_facts(conn, "t1"))
    assert best["algorithm"].value == "XGBoost" and best["algorithm"].mentions == 2
    assert best["algorithm"].source_label == "Proposal.pdf p.2"
    assert list(best) == ["dataset", "algorithm"]               # the blueprint's own field order


def test_ties_go_to_the_earliest_mention_and_all_values_are_listed(conn):
    upload(conn, facts=[fact("algorithm", "LSTM", page=1), fact("algorithm", "ARIMA", page=2)])
    grouped = group_candidates(list_doc_facts(conn, "t1"))
    assert [c.value for c in grouped["algorithm"]] == ["LSTM", "ARIMA"]


def test_the_same_value_in_two_documents_is_one_suggestion(conn):
    upload(conn, "A.pdf", facts=[fact("algorithm", "XGBoost")])
    upload(conn, "B.docx", facts=[fact("algorithm", "XGBoost", page=None)])
    c = best_candidates(list_doc_facts(conn, "t1"))["algorithm"]
    assert c.mentions == 2 and len(c.documents) == 2 and c.filename == "A.pdf"


def test_a_dismissed_value_is_not_suggested_again_even_from_a_new_document(conn):
    upload(conn, "A.pdf", facts=[fact("algorithm", "XGBoost")])
    first = list_doc_facts(conn, "t1")[0]["fact_id"]
    set_facts_status(conn, "t1", [first], "dismissed")
    upload(conn, "B.pdf", facts=[fact("algorithm", "xgboost"), fact("algorithm", "LSTM")])
    assert best_candidates(list_doc_facts(conn, "t1"))["algorithm"].value == "LSTM"


def test_find_group_returns_the_whole_suggestion_or_none(conn):
    upload(conn, facts=[fact("algorithm", "XGBoost"), fact("algorithm", "XGBoost", chunk=2)])
    facts = list_doc_facts(conn, "t1")
    group = find_group(facts, facts[1]["fact_id"])
    assert group is not None and sorted(group.fact_ids) == sorted(f["fact_id"] for f in facts)
    set_facts_status(conn, "t1", group.fact_ids, "accepted")
    assert find_group(list_doc_facts(conn, "t1"), facts[0]["fact_id"]) is None


def test_as_dict_hides_the_text_when_evidence_is_not_allowed(conn):
    upload(conn, facts=[fact("algorithm", "XGBoost", span="use XGBoost")])
    c = best_candidates(list_doc_facts(conn, "t1"))["algorithm"]
    assert c.as_dict()["text"] == "use XGBoost"
    assert "text" not in c.as_dict(include_evidence=False)
