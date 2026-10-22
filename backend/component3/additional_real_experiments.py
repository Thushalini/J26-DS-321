import os
import time
import numpy as np

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from component2.db import connect
from component3.client.c3_reporter import RunReporter

load_dotenv()

TEAM_ID = "live__1d7886d3"
API_URL = "http://127.0.0.1:8000"

EXPERIMENTS = [
    {"name": "short_training", "epochs": 3, "alpha": 0.0001, "seed": 42},
    {"name": "strong_regularization", "epochs": 12, "alpha": 0.1, "seed": 42},
    {"name": "alternate_split", "epochs": 12, "alpha": 0.0001, "seed": 21},
    {"name": "extended_training", "epochs": 25, "alpha": 0.00001, "seed": 42},
]

conn = connect(create_tables=False)
rows = conn.execute("""
    SELECT COALESCE(a.text, ''), s.stage
    FROM stage_labels s
    JOIN activities a ON a.activity_id = s.activity_id
    WHERE s.labeller = 'gold'
""").fetchall()
conn.close()

texts = [row[0] for row in rows]
labels = np.array([row[1] for row in rows])
classes = np.unique(labels)

print(f"Loaded {len(rows)} real labeled records.", flush=True)

for config in EXPERIMENTS:
    print(f"\nStarting: {config['name']}", flush=True)

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=config["seed"],
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    model = SGDClassifier(
        loss="log_loss",
        alpha=config["alpha"],
        random_state=config["seed"],
        learning_rate="optimal",
    )

    reporter = RunReporter(
        API_URL,
        TEAM_ID,
        os.environ.get("C3_API_KEY", ""),
        config["name"],
    )

    for epoch in range(config["epochs"]):
        model.partial_fit(X_train, y_train, classes=classes)

        train_probs = model.predict_proba(X_train)
        val_probs = model.predict_proba(X_val)

        train_loss = log_loss(y_train, train_probs, labels=classes)
        val_loss = log_loss(y_val, val_probs, labels=classes)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))

        reporter.epoch(
            loss=train_loss,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            val_loss=val_loss,
        )

        print(
            f"Epoch {epoch + 1:02d}/{config['epochs']} | "
            f"loss={train_loss:.4f} | "
            f"train={train_acc:.3f} | "
            f"validation={val_acc:.3f}",
            flush=True,
        )

    result = reporter.finish()

    if result:
        print(
            f"RESULT: {config['name']} | "
            f"{result.get('severity_level')} | "
            f"{result.get('recommended_cause')}",
            flush=True,
        )
    else:
        print(f"Submission failed: {config['name']}", flush=True)

    time.sleep(1)

print("\nAll experiments completed.", flush=True)