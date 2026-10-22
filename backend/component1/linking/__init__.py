"""Reply linking: which earlier proposal does a reply ("agreed", "no", "lock it in") answer?

    simulate.py  simulated planning chats where every reply's true parent is known
    features.py  numbers that describe one (reply, candidate proposal) pair
    train.py     train a pairwise classifier and compare it with "the latest proposal" baseline
    predict.py   pick the most likely parent among recent proposals, or None

If the person used the app's Reply button, that link is used directly and no model is needed.
READ BEFORE QUOTING ANY NUMBER: the training chats are SIMULATED (the team chose this over the
Ubuntu IRC corpus). Scores show the method works on the simulation's patterns, not on real chat.
"""
