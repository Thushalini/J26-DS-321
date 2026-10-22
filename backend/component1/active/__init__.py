"""Buddy as an active planning assistant: scores that decide what is worth asking next.

    features.py   the numbers that describe one field right now (shared by every model below)
    simulate.py   simulated planning histories where it is known which fields later change
    risk.py       conflict risk: how likely a field is to be disputed or changed soon (trained)
    train_risk.py trains and evaluates the conflict-risk model against a simple rule

Nothing here replaces conflict detection, plan checks or the feasibility model. These scores only
help Buddy choose WHICH question to ask; Buddy still asks one thing at a time (agent/policy.py).
"""
