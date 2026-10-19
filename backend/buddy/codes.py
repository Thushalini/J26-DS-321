"""Short codes people type: a team's join code and a supervisor's personal code.

Format: 3 capital letters and 3 digits in any order, shown as two groups of three (e.g. H79-KN5).
Typing is forgiving: case, spaces and the dash are ignored.
"""
import re
import secrets

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
LETTER_COUNT = 3
DIGIT_COUNT = 3
GROUP_SIZE = 3
SEPARATOR = "-"
MAX_ATTEMPTS = 50

_random = secrets.SystemRandom()


def _generate():
    chars = [_random.choice(LETTERS) for _ in range(LETTER_COUNT)] + [_random.choice(DIGITS) for _ in range(DIGIT_COUNT)]
    _random.shuffle(chars)
    raw = "".join(chars)
    return SEPARATOR.join(raw[i:i + GROUP_SIZE] for i in range(0, len(raw), GROUP_SIZE))


def normalise(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def code_in_use(conn, code):
    plain = normalise(code)
    return bool(conn.execute("SELECT 1 FROM teams WHERE REPLACE(join_code, '-', '') = ?", (plain,)).fetchone()
                or conn.execute("SELECT 1 FROM users WHERE REPLACE(supervisor_code, '-', '') = ?", (plain,)).fetchone())


def new_code(conn):
    """A code no team and no supervisor already has (team and supervisor codes never collide)."""
    for _ in range(MAX_ATTEMPTS):
        code = _generate()
        if not code_in_use(conn, code):
            return code
    raise RuntimeError("Could not find a free code - try again.")
