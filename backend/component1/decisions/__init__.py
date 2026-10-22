"""Decision records: what the team decided, who proposed it, why, who agreed or objected, and its status.

    tracker.py  create and move decisions from chat messages and accepted document suggestions
    domino.py   when an earlier choice changes, flag the later decisions it may break (pending_fix)
    view.py     decisions as the Blueprint tab shows them, each with its own timeline

Field values (blueprint.fields) say what the plan currently IS. Decisions say how the team GOT
there: one record per proposed value, moving proposed -> confirmed -> finalized, or rejected,
superseded, or pending_fix. Milestones (timeline/milestones.py) are decisions too, with ids "ms:".
"""
