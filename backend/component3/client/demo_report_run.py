"""Demo: train a small REAL model and report it to Component 3 with c3_reporter.

Run from the repository root, with the backend running and these three set in the same terminal:

    $env:C3_URL = "http://127.0.0.1:8000"
    $env:C3_TEAM_ID = "<your team id>"
    $env:C3_API_KEY = "<the C3_API_KEY from backend/.env>"
    python backend/component3/client/demo_report_run.py --scenario overfitting

Scenarios: healthy, overfitting, stalled, premature. The alert appears on that team's Alerts page.
Add --keras to train a Keras network instead (with verbose=0, so nothing is printed) and report it with the
C3Keras callback: this is the case a log reader cannot handle.
(The key is read from the environment so it is never written into a file.)
"""
import argparse
import os
import sys
import warnings

import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from c3_reporter import C3Keras, RunReporter  # noqa: E402

warnings.filterwarnings("ignore")
SCENARIOS = {  # hidden layers, learning rate, epochs, training rows, share of wrong training labels
    "healthy": ((32,), 0.01, 25, None, 0.0),
    "overfitting": ((256, 256), 0.01, 45, 40, 0.25),
    "stalled": ((32,), 1e-6, 25, None, 0.0),
    "premature": ((32,), 0.01, 4, None, 0.0),
}


def keras_run(scenario, url, team, key):
    """Train a small Keras network with verbose=0 and let the callback report it."""
    from tensorflow import keras
    hidden, lr, epochs, n_train, flip = SCENARIOS[scenario]
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X).astype("float32")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    if n_train:
        Xtr, ytr = Xtr[:n_train].copy(), ytr[:n_train].copy()
        wrong = np.random.default_rng(1).random(len(ytr)) < flip
        ytr[wrong] = np.random.default_rng(2).choice(np.unique(y), wrong.sum())
    layers = [keras.layers.Input(shape=(Xtr.shape[1],))]
    layers += [keras.layers.Dense(h, activation="relu") for h in hidden] + [keras.layers.Dense(3, activation="softmax")]
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=0,
              callbacks=[C3Keras(url, team, key, f"keras_{scenario}")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=SCENARIOS, default="overfitting")
    ap.add_argument("--keras", action="store_true", help="train a Keras network and report it with the C3Keras callback")
    a = ap.parse_args()
    url, team, key = os.environ.get("C3_URL"), os.environ.get("C3_TEAM_ID"), os.environ.get("C3_API_KEY")
    if not (url and team and key):
        sys.exit("Set C3_URL, C3_TEAM_ID and C3_API_KEY first (see the top of this file).")
    if a.keras:
        keras_run(a.scenario, url, team, key)
        return
    hidden, lr, epochs, n_train, flip = SCENARIOS[a.scenario]
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    if n_train:
        Xtr, ytr = Xtr[:n_train].copy(), ytr[:n_train].copy()
        wrong = np.random.default_rng(1).random(len(ytr)) < flip
        ytr[wrong] = np.random.default_rng(2).choice(np.unique(y), wrong.sum())
    model = MLPClassifier(hidden_layer_sizes=hidden, learning_rate_init=lr, batch_size=32, random_state=1)
    rep = RunReporter(url, team, key, f"demo_{a.scenario}")
    for _ in range(epochs):
        model.partial_fit(Xtr, ytr, classes=np.unique(y))
        rep.epoch(loss=log_loss(ytr, model.predict_proba(Xtr), labels=np.unique(y)),
                  train_accuracy=model.score(Xtr, ytr), val_accuracy=model.score(Xva, yva))
    out = rep.finish()
    if out:
        print("Assumed (not measured):", ", ".join(out.get("assumed", [])))


if __name__ == "__main__":
    main()
