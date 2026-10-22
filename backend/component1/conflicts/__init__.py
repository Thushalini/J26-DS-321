"""Conflicts by meaning: are two values for one field the same choice, or a real disagreement?

    nli.py   a pretrained natural language inference model (DeBERTa-v3 small cross-encoder)
    same.py  same_value(field, a, b): known names first, then NLI for free-text fields

The NLI model is strong at spotting CONTRADICTIONS ("tourist arrivals" vs "hotel occupancy": 1.00)
and weak at recognising paraphrases ("churn" vs "whether a customer leaves" came out as a
contradiction), so it is used the way the proposal describes: two free-text values count as the
same only when NLI finds no contradiction either way AND one implies the other. Paraphrases with
no shared meaning in the words stay a disagreement; that is a known limit to report.
"""
