"""Tests for component1/labeling/import_chat.py and kappa.py.

The chat lines below are made-up fixtures that only test the parser's formats.
They are not data and must never be added to your dataset.
"""
import csv
import math

from component1.labeling.import_chat import (
    build_rows,
    is_skippable,
    parse_chat,
    parse_ts,
    redact,
    write_outputs,
)
from component1.labeling.kappa import cohen_kappa, interpret, read_labels

ANDROID = [
    "14/03/2026, 10:02 - Messages and calls are end-to-end encrypted.\n",
    "14/03/2026, 10:03 - Alice Perera: I suggest we use XGBoost\n",
    "14/03/2026, 10:04 - Bob Silva: I agree with Alice\n",
    "and I can write the notes\n",
    "14/03/2026, 10:05 - Alice Perera: <Media omitted>\n",
]

IOS = [
    "\u200e[14/03/2026, 10:02:15 AM] Alice Perera: let's finalize it\n",
    "[14/03/2026, 10:03:01 AM] Bob Silva: ok\n",
]


# ------------------------------------------------------------- parsing
def test_parse_android_format_and_continuation_lines():
    msgs = parse_chat(ANDROID)
    assert [m["speaker"] for m in msgs] == ["Alice Perera", "Bob Silva", "Alice Perera"]
    assert msgs[1]["text"] == "I agree with Alice\nand I can write the notes"


def test_system_lines_are_dropped():
    msgs = parse_chat(ANDROID)
    assert all("encrypted" not in m["text"] for m in msgs)


def test_parse_ios_format_with_invisible_marks():
    msgs = parse_chat(IOS)
    assert len(msgs) == 2
    assert msgs[0]["speaker"] == "Alice Perera"


def test_media_messages_are_skipped():
    assert is_skippable("<Media omitted>")
    assert not is_skippable("let's use RMSE")


def test_parse_ts_dayfirst_and_monthfirst():
    assert parse_ts("14/03/2026", "10:02").day == 14
    assert parse_ts("03/14/2026", "10:02", dayfirst=False).day == 14
    assert parse_ts("14/03/2026", "10:02 PM").hour == 22


# ------------------------------------------------------- anonymisation
def test_redact_masks_names_links_emails_and_phones():
    speaker_map = {"Alice Perera": "Member_A", "Bob Silva": "Member_B"}
    text = "Alice said mail bob@example.com or call +94 77 123 4567, see https://x.com/a"
    out = redact(text, speaker_map)
    assert "Alice" not in out
    assert "bob@example.com" not in out
    assert "123 4567" not in out
    assert "https://x.com" not in out
    assert "Member_A" in out


def test_build_rows_anonymises_and_adds_previous_message():
    msgs = parse_chat(ANDROID)
    speaker_map, rows = build_rows(msgs)
    assert set(speaker_map.values()) == {"Member_A", "Member_B"}
    assert len(rows) == 2  # the media message was skipped
    assert rows[0]["prev_message"] == ""
    assert rows[1]["prev_message"].startswith("Member_A:")
    assert "Alice" not in rows[1]["message"]


def test_write_outputs_creates_both_rater_sheets(tmp_path):
    speaker_map, rows = build_rows(parse_chat(ANDROID))
    write_outputs(rows, speaker_map, tmp_path)
    for name in ("messages_anonymised.csv", "sheet_rater_a.csv", "sheet_rater_b.csv", "speaker_key.json"):
        assert (tmp_path / name).exists()
    with open(tmp_path / "sheet_rater_a.csv", newline="", encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    assert sheet[0]["intent"] == ""  # blank, for the rater to fill in


# --------------------------------------------------------------- kappa
def test_kappa_known_textbook_value():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> observed 0.70, expected 0.50, kappa 0.40
    a = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert math.isclose(cohen_kappa(a, b), 0.4, abs_tol=1e-9)


def test_kappa_perfect_agreement_is_one():
    a = ["proposal", "agreement", "other", "finalization"]
    assert math.isclose(cohen_kappa(a, list(a)), 1.0)


def test_kappa_single_label_is_undefined():
    assert math.isnan(cohen_kappa(["other"] * 5, ["other"] * 5))


def test_kappa_rejects_bad_input():
    for bad in (([], []), (["a"], ["a", "b"])):
        try:
            cohen_kappa(*bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_interpretation_bands():
    assert interpret(0.75) == "substantial"
    assert interpret(0.85) == "almost perfect"
    assert interpret(-0.1).startswith("poor")

def test_read_labels_handles_excel_bom(tmp_path):
    path = tmp_path / "sheet.csv"
    path.write_text("msg_id,intent\nM1,Proposal\nM2,\n", encoding="utf-8-sig")  # BOM, like Excel
    assert read_labels(path, "msg_id", "intent") == {"M1": "proposal"}  # blank row skipped
