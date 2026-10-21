"""Reads a SLIIT "Student Performance Profile" PDF into module codes, credits and grades.

Only what the competency score needs is kept: no name, no registration number, no GPA.
A module taken more than once counts by its latest attempt.
"""
import io
import re

from pypdf import PdfReader

ROW = re.compile(r"^(IT\d{4})\s+(.+?)\s+(\d+)\s+(\d+)\s+([A-F][+-]?|F\(\w[+-]?\))\s*$")


class SheetError(ValueError):
    pass


def read_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as error:
        raise SheetError("This file isn't a readable PDF.") from error


def parse_text(text: str) -> list[dict]:
    latest = {}
    for line in text.splitlines():
        match = ROW.match(line.strip())
        if not match:
            continue
        code, name, credits, attempt, grade = match.groups()
        grade = "F" if grade.startswith("F(") else grade
        if code not in latest or int(attempt) >= latest[code]["attempt"]:
            latest[code] = {"code": code, "name": name, "credits": int(credits), "attempt": int(attempt), "grade": grade}
    if not latest:
        raise SheetError("No module results found. Is this a Student Performance Profile with selectable text?")
    return [{k: v for k, v in row.items() if k != "attempt"} for row in latest.values()]


def parse_sheet(pdf_bytes: bytes) -> list[dict]:
    return parse_text(read_text(pdf_bytes))
