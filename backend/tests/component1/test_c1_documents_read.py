"""Tests for component1/documents/parse.py, chunk.py and extract.py. Offline; the text is invented."""
import json

import pytest

from component1.documents.chunk import Chunk, chunk_pages
from component1.documents.extract import extract_document, max_llm_chunks_from_env
from component1.documents.parse import MAX_BYTES, DocumentError, Page, kind_of, parse_document

from c1_docfiles import make_docx, make_pdf


# ----------------------------------------------------------------- parsing
def test_pdf_pages_keep_their_numbers():
    data = make_pdf([["Project proposal"], ["We will use the Kaggle tourism dataset."]])
    pages = parse_document("Proposal.pdf", data)
    assert [p.number for p in pages] == [1, 2]
    assert "Kaggle tourism dataset" in pages[1].text


def test_word_paragraphs_and_tables_are_read_as_one_page():
    data = make_docx(["Our goal is to reduce hotel overbooking."], table=[["Metric", "RMSE"]])
    (page,) = parse_document("Plan.DOCX", data)
    assert page.number is None
    assert "reduce hotel overbooking" in page.text and "Metric | RMSE" in page.text


def test_text_and_markdown_are_read_with_or_without_a_bom():
    with_bom = b"\xef\xbb\xbf" + "Let's use XGBoost".encode("utf-8")
    assert parse_document("notes.md", with_bom)[0].text == "Let's use XGBoost"
    assert parse_document("notes.txt", "caf\xe9 plan".encode("cp1252"))[0].text == "caf\xe9 plan"


@pytest.mark.parametrize("name, data, words", [
    ("slides.pptx", b"x", "not supported"),
    ("plan.pdf", b"", "empty"),
    ("plan.pdf", b"not a pdf at all", "could not be read"),
    ("plan.docx", b"not a docx", "could not be read"),
    ("plan.txt", b"   \n  ", "No text found"),
])
def test_unusable_files_give_a_message_for_the_uploader(name, data, words):
    with pytest.raises(DocumentError, match=words):
        parse_document(name, data)


def test_a_pdf_without_a_text_layer_says_it_may_be_scanned():
    with pytest.raises(DocumentError, match="scanned"):
        parse_document("scan.pdf", make_pdf([[]]))


def test_files_over_the_size_limit_are_refused():
    with pytest.raises(DocumentError, match="larger than"):
        parse_document("big.txt", b"a" * (MAX_BYTES + 1))


def test_kind_of_ignores_case():
    assert kind_of("A.PDF") == "pdf" and kind_of("b.Markdown") == "text"


# ---------------------------------------------------------------- chunking
def test_paragraphs_are_kept_together_and_pages_are_kept():
    pages = [Page(1, "First paragraph about the plan.\n\nSecond paragraph about data."), Page(2, "Third one on page two.")]
    chunks = chunk_pages(pages, max_chars=1000)
    assert [(c.page, c.text) for c in chunks] == [
        (1, "First paragraph about the plan. Second paragraph about data."),
        (2, "Third one on page two."),
    ]
    assert [c.number for c in chunks] == [1, 2]


def test_long_text_is_cut_between_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_pages([Page(1, text)], max_chars=120)
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    assert " ".join(c.text for c in chunks) == text


def test_one_huge_sentence_is_still_cut():
    chunks = chunk_pages([Page(1, "word " * 400)], max_chars=100)
    assert chunks and all(len(c.text) <= 100 for c in chunks)


@pytest.mark.parametrize("heading", ["References", "10. REFERENCES LIST", "Bibliography:", "Works Cited"])
def test_reading_stops_at_the_reference_list(heading):
    pages = [Page(1, f"We will use XGBoost for this task.\n{heading}\n[1] Smith used SVM on other data."),
             Page(2, "Appendix that comes after the references.")]
    text = " ".join(c.text for c in chunk_pages(pages))
    assert "XGBoost" in text and "SVM" not in text and "Appendix" not in text


def test_a_table_of_contents_line_does_not_stop_reading():
    chunks = chunk_pages([Page(1, "10. REFERENCES LIST ........ 41\n\nWe will use XGBoost for the model.")])
    assert any("XGBoost" in c.text for c in chunks)


def test_words_broken_across_lines_are_joined_and_tiny_pieces_dropped():
    chunks = chunk_pages([Page(1, "A gradient boosted classi-\nfier will be trained.\n\n12")])
    assert [c.text for c in chunks] == ["A gradient boosted classifier will be trained."]


# -------------------------------------------------------------- extraction
class CountingLLM:
    def __init__(self, reply='{"items": []}'):
        self.reply, self.calls = reply, 0

    def __call__(self, prompt):
        self.calls += 1
        return self.reply


def test_rules_find_facts_with_their_page_and_exact_words():
    chunks = [Chunk(1, 3, "We will use the Kaggle tourism dataset and train XGBoost.")]
    facts = extract_document(chunks, fallback_llm=None)
    found = {(f.field, f.value, f.page, f.method) for f in facts}
    assert ("algorithm", "XGBoost", 3, "rules") in found
    assert ("dataset", "Kaggle tourism dataset", 3, "rules") in found
    assert all(f.span in chunks[0].text for f in facts)


def test_the_language_model_is_only_used_where_the_rules_found_nothing():
    reply = json.dumps({"items": [{"field": "objective", "value": "help hotels plan staff",
                                   "span": "help hotels plan staff"}]})
    llm = CountingLLM(reply)
    chunks = [Chunk(1, 1, "Let's use XGBoost for the model."),               # rules: no call
              Chunk(2, 1, "The model should help hotels plan staff."),       # planning words, no rule: call
              Chunk(3, 1, "Thanks to our families for their support.")]      # no planning words: no call
    facts = extract_document(chunks, fallback_llm=llm, max_llm_chunks=5)
    assert llm.calls == 1
    assert {(f.field, f.method) for f in facts} == {("algorithm", "rules"), ("objective", "llm")}


def test_the_language_model_calls_are_capped_per_document():
    llm = CountingLLM()
    chunks = [Chunk(i, 1, f"Section {i}: the model should help hotels plan.") for i in range(1, 9)]
    extract_document(chunks, fallback_llm=llm, max_llm_chunks=3)
    assert llm.calls == 3


def test_a_failing_language_model_skips_the_chunk():
    def broken(_prompt):
        raise TimeoutError("slow")
    facts = extract_document([Chunk(1, 1, "The model should help hotels plan.")], fallback_llm=broken)
    assert facts == []


def test_the_cap_can_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("C1_DOC_MAX_LLM_CHUNKS", "4")
    assert max_llm_chunks_from_env() == 4
    monkeypatch.setenv("C1_DOC_MAX_LLM_CHUNKS", "lots")
    assert max_llm_chunks_from_env() == 10
    monkeypatch.delenv("C1_DOC_MAX_LLM_CHUNKS")
    assert max_llm_chunks_from_env() == 10


def test_a_whole_pdf_goes_from_bytes_to_facts():
    data = make_pdf([["Introduction text that sets the scene."],
                     ["We will predict tourist arrivals using ARIMA.", "We report RMSE."],
                     ["References", "[1] A paper that used LSTM."]])
    facts = extract_document(chunk_pages(parse_document("p.pdf", data)), fallback_llm=None)
    by_field = {f.field: (f.value, f.page) for f in facts}
    assert by_field["algorithm"] == ("ARIMA", 2)
    assert by_field["target_variable"][1] == 2
    assert "LSTM" not in {f.value for f in facts}


# ------------------------------------- cases found on a real proposal's text
def test_each_sentence_is_read_on_its_own_so_a_paragraph_keeps_all_its_options():
    chunk = Chunk(1, 2, "We tried XGBoost first. The final model might use Random Forest. We report RMSE.")
    facts = extract_document([chunk], fallback_llm=None)
    algos = [(f.value, f.hedged) for f in facts if f.field == "algorithm"]
    assert algos == [("XGBoost", False), ("Random Forest", True)]      # only the unsure sentence is hedged
    assert ("evaluation_metric", "RMSE") in {(f.field, f.value) for f in facts}


@pytest.mark.parametrize("text, value", [
    ("Records from the PIPES meta-dataset are used.", "PIPES meta-dataset"),
    ("It is validated against the AMI Meeting Corpus as a check.", "AMI Meeting Corpus"),
    ("Labels come from the Switchboard Dialogue Act Corpus for training.", "Switchboard Dialogue Act Corpus"),
])
def test_named_corpora_and_meta_datasets_are_datasets(text, value):
    facts = extract_document([Chunk(1, 1, text)], fallback_llm=None)
    assert [f.value for f in facts if f.field == "dataset"] == [value]


@pytest.mark.parametrize("text", ["The main data needed is the team's chat.", "We build a meta-dataset later.",
                                  "Read the corpus first."])
def test_generic_talk_about_data_is_not_a_dataset(text):
    assert not [f for f in extract_document([Chunk(1, 1, text)], fallback_llm=None) if f.field == "dataset"]


def test_cohens_kappa_is_a_metric():
    facts = extract_document([Chunk(1, 1, "Agreement is measured with Cohen's Kappa.")], fallback_llm=None)
    assert [(f.field, f.value) for f in facts] == [("evaluation_metric", "Cohen's Kappa")]
