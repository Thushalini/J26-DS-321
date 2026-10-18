#!/usr/bin/env python3
"""Turn a WhatsApp-style text export into anonymised labeling sheets.

Standard library only. Usage (run from the backend folder):

    python -m component1.labeling.import_chat component1/data/raw/team_chat.txt --out component1/data/labeling_out

It writes into the --out folder:
    messages_anonymised.csv   all kept messages, speakers replaced by Member_A, Member_B...
    sheet_rater_a.csv         sheet for rater A (intent column to fill in)
    sheet_rater_b.csv         identical sheet for rater B (so raters label independently)
    speaker_key.json          real name -> label. PRIVATE. Never share or commit it.

Dates are read as day/month/year. Use --monthfirst for month/day/year exports.
Names inside message text are replaced when they match a speaker's name, but
ALWAYS read the output once yourself and remove anything else that identifies a person.
"""
import argparse
import csv
import json
import re
import string
import sys
from datetime import datetime
from pathlib import Path

TS_START = re.compile(
    r"^\[?(\d{1,2}[/.]\d{1,2}[/.]\d{2,4}),?\s+"
    r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)\]?\s*(?:-\s*)?(.*)$"
)
SENDER = re.compile(r"^([^:]{1,60}?):\s(.*)$")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<![\w/])\+?\d[\d\s-]{7,}\d(?![\w/])")

INTENT_LABELS = "proposal / agreement / finalization / rejection / other"


def parse_chat(lines):
    """Return a list of {date, time, speaker, text}. System lines are dropped."""
    messages = []
    current = None
    for raw in lines:
        line = raw.rstrip("\n").replace("\u200e", "").replace("\u200f", "").lstrip("\ufeff")
        m = TS_START.match(line)
        if m:
            date_s, time_s, rest = m.groups()
            s = SENDER.match(rest)
            if s:
                current = {
                    "date": date_s,
                    "time": time_s,
                    "speaker": s.group(1).strip(),
                    "text": s.group(2),
                }
                messages.append(current)
            else:
                current = None  # a system line such as "messages are encrypted"
        elif current is not None and line.strip():
            current["text"] += "\n" + line  # a continuation line of a longer message
    return messages


def is_skippable(text):
    t = text.strip().lower()
    return (
        not t
        or "omitted" in t
        or t in {"null", "this message was deleted", "you deleted this message"}
    )


def speaker_label(index):
    letters = string.ascii_uppercase
    return f"Member_{letters[index]}" if index < 26 else f"Member_{index + 1}"


def build_speaker_map(messages):
    mapping = {}
    for m in messages:
        if m["speaker"] not in mapping:
            mapping[m["speaker"]] = speaker_label(len(mapping))
    return mapping


def redact(text, speaker_map):
    """Mask links, emails, phone-like numbers and speaker names."""
    text = URL.sub("[URL]", text)
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    names = []
    for real, label in speaker_map.items():
        names.append((real, label))
        for token in real.split():
            if len(token) >= 3:
                names.append((token, label))
    for name, label in sorted(names, key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{re.escape(name)}\b", label, text, flags=re.IGNORECASE)
    return text


def parse_ts(date_s, time_s, dayfirst=True):
    date_s = date_s.replace(".", "/")
    time_s = re.sub(r"\s+", " ", time_s.strip()).replace(".", "")
    date_formats = ["%d/%m/%Y", "%d/%m/%y"] if dayfirst else ["%m/%d/%Y", "%m/%d/%y"]
    time_formats = ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"]
    for dfmt in date_formats:
        for tfmt in time_formats:
            try:
                return datetime.strptime(f"{date_s} {time_s}", f"{dfmt} {tfmt}")
            except ValueError:
                continue
    return None


def build_rows(messages, dayfirst=True, context=1):
    """Return (speaker_map, rows). Each row has the previous message(s) for context."""
    kept = [m for m in messages if not is_skippable(m["text"])]
    speaker_map = build_speaker_map(kept)
    rows = []
    for i, m in enumerate(kept):
        ts = parse_ts(m["date"], m["time"], dayfirst)
        prev = []
        for j in range(max(0, i - context), i):
            p = kept[j]
            prev.append(f"{speaker_map[p['speaker']]}: {redact(p['text'], speaker_map)}")
        rows.append(
            {
                "msg_id": f"M{i + 1:04d}",
                "timestamp": ts.isoformat(sep=" ") if ts else f"{m['date']} {m['time']}",
                "speaker": speaker_map[m["speaker"]],
                "prev_message": " || ".join(prev),
                "message": redact(m["text"], speaker_map),
            }
        )
    return speaker_map, rows


def write_outputs(rows, speaker_map, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "messages_anonymised.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["msg_id", "timestamp", "speaker", "prev_message", "message"])
        w.writeheader()
        w.writerows(rows)

    sheet_fields = ["msg_id", "speaker", "prev_message", "message", "intent", "unsure", "notes"]
    for rater in ("a", "b"):
        with open(out / f"sheet_rater_{rater}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sheet_fields)
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        "msg_id": r["msg_id"],
                        "speaker": r["speaker"],
                        "prev_message": r["prev_message"],
                        "message": r["message"],
                        "intent": "",
                        "unsure": "",
                        "notes": "",
                    }
                )

    with open(out / "speaker_key.json", "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, indent=2, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chat_file", help="exported chat .txt file")
    ap.add_argument("--out", default="component1/data/labeling_out", help="output folder (default: component1/data/labeling_out)")
    ap.add_argument("--monthfirst", action="store_true", help="dates are month/day/year")
    ap.add_argument("--context", type=int, default=1, help="how many previous messages to show (default 1)")
    args = ap.parse_args(argv)

    with open(args.chat_file, encoding="utf-8") as f:
        messages = parse_chat(f)
    speaker_map, rows = build_rows(messages, dayfirst=not args.monthfirst, context=args.context)
    write_outputs(rows, speaker_map, args.out)

    print(f"Read {len(messages)} messages, kept {len(rows)}, {len(speaker_map)} speakers.")
    print(f"Wrote files to {args.out}/")
    print("Intent labels for the sheets:", INTENT_LABELS)
    print("Now read messages_anonymised.csv and remove anything that still identifies a person.")
    print("Keep speaker_key.json private.")


if __name__ == "__main__":
    sys.exit(main())