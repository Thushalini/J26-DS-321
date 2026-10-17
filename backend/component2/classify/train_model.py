"""
Step 2d: The main lifecycle classifier (FR04).

1. Turn every activity's text into a 384-number "meaning vector" with the
   all-MiniLM-L6-v2 sentence-transformer (texts with similar meaning get
   similar vectors, even with different words).
2. Train a Logistic Regression on the gold-labelled vectors.
3. Test it fairly with 5-fold cross-validation and compare with the keyword
   baseline on exactly the same items (macro F1, target >= 0.75).
4. Label ALL activities with stage + confidence and save as labeller 'model'.

Run from backend:
    python -m component2.classify.train_model
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from component2.classify.keyword_baseline import PATTERNS
from component2.config import LABEL_DIR, MODEL_DIR
from component2.db import connect, save_labels

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHARS = 1000            # MiniLM reads ~256 words; longer text is cut anyway


def model_input(source, text):
    return f"[{source}] {text[:MAX_CHARS]}"     # telling the model where text came from helps it


def get_embeddings(conn):
    """Embed every activity once and cache the result, so re-runs take seconds."""
    rows = conn.execute("SELECT activity_id, source, text FROM activities ORDER BY activity_id").fetchall()
    ids = [r[0] for r in rows]
    cache = MODEL_DIR / "embeddings.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["ids"]) == ids:
            print(f"Using cached embeddings for {len(ids)} activities")
            return ids, data["vectors"]

    from sentence_transformers import SentenceTransformer
    print(f"Embedding {len(ids)} activities with {MODEL_NAME} (first time takes a few minutes)...")
    encoder = SentenceTransformer(MODEL_NAME)
    vectors = encoder.encode([model_input(s, t) for _, s, t in rows],
                             batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids, dtype=object), vectors=vectors)
    return ids, vectors


SOURCES = ("commit", "notebook_cell", "chat")


def structure_features(conn, ids):
    """Small extra signals the text embedding cannot see well:
    where the text came from, whether a notebook cell is markdown, and the keyword-clue counts."""
    info = {aid: (source, json.loads(meta or "{}").get("cell_type"), text)
            for aid, source, meta, text in conn.execute("SELECT activity_id, source, meta, text FROM activities").fetchall()}
    rows = []
    for aid in ids:
        source, cell_type, text = info[aid]
        hits = [sum(1 for p in pats if p.search(text)) for pats in PATTERNS.values()]
        rows.append([source == s for s in SOURCES] + [cell_type == "markdown", len(text) < 40] + hits)
    return np.array(rows, dtype=float)


def make_classifier():
    """Scale features, then Logistic Regression; the regularisation strength C is chosen
    by an inner 3-fold search on the training part of each fold (so the test fold stays unseen)."""
    pipe = Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    return GridSearchCV(pipe, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, scoring="f1_macro",
                        cv=StratifiedKFold(3, shuffle=True, random_state=0))


def main():
    conn = connect()
    ids, vectors = get_embeddings(conn)
    row_of = {aid: n for n, aid in enumerate(ids)}
    extra = structure_features(conn, ids)

    gold = dict(conn.execute("SELECT activity_id, stage FROM stage_labels WHERE labeller='gold'").fetchall())
    baseline = dict(conn.execute("SELECT activity_id, stage FROM stage_labels WHERE labeller='baseline'").fetchall())
    gold_ids = [i for i in gold if i in row_of]
    if len(gold_ids) < 50:
        print(f"Only {len(gold_ids)} gold labels - run load_ratings first.")
        return

    rows_gold = [row_of[i] for i in gold_ids]
    X_text = vectors[rows_gold]                                   # MiniLM only
    X_full = np.hstack([vectors, extra])                          # MiniLM + structure signals
    y = np.array([gold[i] for i in gold_ids])

    # Fair test: each item is predicted by a model that never saw it
    n_splits = max(2, min(5, int(np.unique(y, return_counts=True)[1].min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_base = np.array([baseline.get(i, "other") for i in gold_ids])
    y_text = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), X_text, y, cv=cv)
    y_pred = cross_val_predict(make_classifier(), X_full[rows_gold], y, cv=cv)

    print(f"\nGold items: {len(gold_ids)}   ({n_splits}-fold cross-validation)")
    print(f"Macro F1  keyword baseline              : {f1_score(y, y_base, average='macro', zero_division=0):.3f}")
    print(f"Macro F1  MiniLM + LogReg (text only)   : {f1_score(y, y_text, average='macro', zero_division=0):.3f}")
    print(f"Macro F1  MiniLM + structure + tuned C  : {f1_score(y, y_pred, average='macro', zero_division=0):.3f}   (target >= 0.75)")
    print(f"Accuracy  final model                   : {np.mean(y_pred == y):.3f}")

    # Round 2 items were chosen on purpose (rare stages), so also report the score on the
    # round 1 items only - that sample was random, so it is the fair, unbiased estimate.
    import csv
    r1_file = LABEL_DIR / "rater1.csv"
    if r1_file.exists() and (LABEL_DIR / "rater1_round2.csv").exists():
        with open(r1_file, newline="", encoding="utf-8-sig") as f:
            round1 = {row["activity_id"] for row in csv.DictReader(f)}
        mask = np.array([i in round1 for i in gold_ids])
        if mask.any():
            print(f"Macro F1  final model, round-1 items only (random sample): "
                  f"{f1_score(y[mask], y_pred[mask], average='macro', zero_division=0):.3f}  ({mask.sum()} items)")

    # Human ceiling: how well does one rater "predict" the other? A model can hardly beat this.
    r1 = dict(conn.execute("SELECT activity_id, stage FROM stage_labels WHERE labeller='rater1'").fetchall())
    r2 = dict(conn.execute("SELECT activity_id, stage FROM stage_labels WHERE labeller='rater2'").fetchall())
    both = [i for i in r1 if i in r2]
    if both:
        human = f1_score([r1[i] for i in both], [r2[i] for i in both], average="macro", zero_division=0)
        print(f"Macro F1  rater 1 vs rater 2 (human ceiling): {human:.3f}")
    print("\nPer-stage results for the final model:")
    print(classification_report(y, y_pred, zero_division=0))

    # Final model: train on all gold labels, then label every activity
    clf = make_classifier().fit(X_full[rows_gold], y)
    print(f"Chosen C: {clf.best_params_['clf__C']}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "stage_classifier.joblib")
    probs = clf.predict_proba(X_full)
    stages = clf.classes_[probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    save_labels(conn, "model", [(ids[n], stages[n], round(float(conf[n]), 3)) for n in range(len(ids))])
    low = int((conf < 0.5).sum())
    print(f"Labelled all {len(ids)} activities. Low-confidence (< 0.5): {low} -> candidates for LLM check")


if __name__ == "__main__":
    main()