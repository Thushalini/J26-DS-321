"""Uploaded project documents: read them, find planning facts, and offer those facts to the team.

    parse.py       file bytes -> pages of text (PDF, Word, plain text, Markdown)
    chunk.py       pages -> short chunks with their page number, stopping at the reference list
    extract.py     chunks -> facts (rules first, the language model only when they find nothing)
    candidates.py  facts -> one suggested value per field
    integrate.py   an accepted suggestion -> blueprint events with document provenance

Document facts are SUGGESTIONS, not blueprint values. Documents discuss alternatives ("earlier
work used SVM, CNN and LSTM"), so writing every mention straight into the blueprint would invent
conflicts. A suggestion becomes part of the plan only when a team member accepts it, and then it
counts as two sources agreeing: the document and that member.
"""
