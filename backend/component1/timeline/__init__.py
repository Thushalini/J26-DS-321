"""The project's timeline: its overall deadline or length, and its named milestones.

    find.py   find timeline phrases in a sentence ("we have 8 weeks", "submission is on 15 Dec")
    dates.py  turn a stored value into dates ("8 weeks" said on 1 Oct -> ends 26 Nov, 40 days left)

The blueprint's `timeline` field holds ONE value: the overall deadline or the project length.
Named milestones (demo, presentation, mid evaluation, ...) are recorded as decisions on the
timeline field, so a demo date and a submission date never look like a disagreement.
"""
