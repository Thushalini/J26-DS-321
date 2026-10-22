import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

from component2.db import connect
from component3.client.c3_reporter import RunReporter

TEAM_ID = "live__1d7886d3"
API_URL = "http://127.0.0.1:8000"

conn = connect(create_tables=False)
rows = conn.execute("""
    SELECT COALESCE(a.text, ''), s.stage
    FROM stage_labels s
    JOIN activities a ON a.activity_id = s.activity_id
    WHERE s.labeller = 'gold'
""").fetchall()
conn.close()

texts = [text for text, label in rows]
labels = np.array([label for text, label in rows])
classes = np.unique(labels)

X_train_text, X_val_text, y_train, y_val = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

model = SGDClassifier(loss="log_loss", random_state=42, learning_rate="optimal")

reporter = RunReporter(
    API_URL,
    TEAM_ID,
    os.environ.get("C3_API_KEY", ""),
    "real_gold_labels"
)

print(f"Training on {len(y_train)} real records; validating on {len(y_val)}.")
print("Starting 12 genuine training epochs...")

for epoch in range(12):
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
        f"Epoch {epoch + 1:02d}/12 | "
        f"loss={train_loss:.4f} | "
        f"train_acc={train_acc:.3f} | "
        f"val_acc={val_acc:.3f}"
    )

print("Training finished.")
result = reporter.finish()

if result is None:
    print("Component 3 submission did not succeed. Training measurements were real.")
else:
    print("Component 3 analysis:", result)
