"""Shared setup for Component 2's tests.

conn        a throwaway SQLite database (never the shared one)
add         saves activities (and their stage labels) into it in one call
fake_embed  stands in for MiniLM: same words -> same vector, no shared words -> similarity 0,
            so the tests are fast, need no model download and give the same result every time
"""
import zlib

import numpy as np
import pytest

DIMS = 512


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    import component2.config as c2config
    import component2.db as c2db

    if c2config.DATABASE_URL:
        pytest.skip("Refusing to run against a real database: DATABASE_URL is set.")
    monkeypatch.setattr(c2db, "DB_PATH", tmp_path / "c2_test.db")
    c = c2db.connect()
    yield c
    c.close()


@pytest.fixture()
def add(conn):
    """add(team_id, [(activity_id, member_id, source, ts, text, stage), ...], labeller="model", meta=None)
    stage None = leave the activity unlabelled."""
    from component2.db import save_activities, save_labels

    def _add(team_id, rows, labeller="model", meta=None):
        save_activities(conn, [{"activity_id": aid, "team_id": team_id, "source": source, "member_id": member,
                                "ts": ts, "text": text, "meta": (meta or {}).get(aid, {})}
                               for aid, member, source, ts, text, _ in rows])
        save_labels(conn, labeller, [(aid, stage, 0.9) for aid, _, _, _, _, stage in rows if stage])
    return _add


def _vector(text):
    v = np.zeros(DIMS)
    for word in text.lower().split():
        v[zlib.crc32(word.encode()) % DIMS] += 1
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def embed_words(texts):
    return np.array([_vector(t) for t in texts])


@pytest.fixture()
def fake_embed():
    return embed_words
