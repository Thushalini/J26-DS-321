"""Notebook log reader: find real training runs in a team's Jupyter notebooks.

Most student projects print their training progress ("Epoch 3/20 ... loss: 0.41 - val_accuracy: 0.83")
and Jupyter saves that text inside the .ipynb file. This reader finds those logs and turns each training
loop into one run (epochs with train accuracy, validation accuracy and loss) that Component 3 can score.

Reads: Keras (versions 2 and 3), PyTorch-style prints ("Epoch [3/20], Loss: 0.41, Val Acc: 83.2%"),
and similar logs that name the metrics loss / accuracy / val_loss / val_accuracy (or acc, val_acc).
A loop is only used when it has at least 3 epochs and shows loss and BOTH training and validation accuracy.

    python -m backend.component3.collector.notebook_logs path/to/notebook.ipynb [--score]

GPU use is not in a notebook, so a neutral value is assumed and the output says so.
"""
import argparse
import json
import re
from pathlib import Path

NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
EPOCH_RE = re.compile(r"epoch[\s:\[\(#]*(\d+)\s*(?:/|of)?\s*(\d+)?", re.IGNORECASE)
METRIC_RE = re.compile(r"([A-Za-z][A-Za-z _\-\.]{0,24}?)\s*[:=]\s*" + NUM + r"\s*(%?)")
MIN_EPOCHS = 3


_NAMES = {
    "loss": "loss", "trainloss": "loss", "trainingloss": "loss",
    "valloss": "val_loss", "validationloss": "val_loss", "validloss": "val_loss", "testloss": "val_loss",
    "accuracy": "train_accuracy", "acc": "train_accuracy", "trainacc": "train_accuracy",
    "trainaccuracy": "train_accuracy", "trainingaccuracy": "train_accuracy", "trainingacc": "train_accuracy",
    "valaccuracy": "val_accuracy", "valacc": "val_accuracy", "validationaccuracy": "val_accuracy",
    "validacc": "val_accuracy", "testaccuracy": "val_accuracy", "testacc": "val_accuracy",
}


def _key(name):
    """'ms/step - val_accuracy' -> 'val_accuracy': try the last three words, two, then one."""
    words = re.findall(r"[A-Za-z]+", name.lower())
    for n in (3, 2, 1):
        if len(words) >= n and "".join(words[-n:]) in _NAMES:
            return _NAMES["".join(words[-n:])]
    return None


def _output_texts(cell):
    """Text a code cell printed (stdout / stderr streams and plain-text results)."""
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            t = out.get("text", "")
            parts.append("".join(t) if isinstance(t, list) else t)
        elif out.get("output_type") in ("execute_result", "display_data"):
            t = out.get("data", {}).get("text/plain", "")
            parts.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(parts)


def _segments(text):
    """Split a cell's text into per-epoch pieces; a restarting epoch counter starts a new run."""
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)            # colour codes in Keras 3 output
    text = text.replace("\x08", "").replace("\r", "\n")
    marks = [(m.start(), int(m.group(1))) for m in EPOCH_RE.finditer(text)]
    runs, cur, prev_epoch = [], [], 0
    for i, (pos, ep) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        if ep < prev_epoch and cur:                        # counter went back: a new training loop
            runs.append(cur); cur = []
        cur.append((ep, text[pos:end]))
        prev_epoch = ep
    if cur:
        runs.append(cur)
    return runs


def _metrics(segment):
    """The LAST value of each metric in an epoch's text (progress bars update the numbers as they go)."""
    found = {}
    for name, value, pct in METRIC_RE.findall(segment):
        k = _key(name)
        if not k:
            continue
        v = float(value)
        if k.endswith("accuracy") and (pct == "%" or v > 1.5):
            v /= 100.0
        found[k] = v
    return found


def parse_text(text):
    """-> (runs, skipped). Each run: a list of epoch dicts."""
    runs, skipped = [], []
    for seg_run in _segments(text):
        epochs = {}
        for ep, seg in seg_run:
            m = _metrics(seg)
            if m:
                epochs.setdefault(ep, {}).update(m)          # several prints for one epoch: the last values win
        rows = [{"epoch": e, **m} for e, m in sorted(epochs.items())]
        good = [r for r in rows if "loss" in r and "train_accuracy" in r and "val_accuracy" in r]
        if len(good) >= MIN_EPOCHS and len(good) >= 0.8 * len(rows):
            runs.append(good)
        elif rows:
            missing = [k for k in ("loss", "train_accuracy", "val_accuracy") if not any(k in r for r in rows)]
            skipped.append(f"{len(rows)} epochs found but " + (f"no {', '.join(missing)}" if missing else "too few complete epochs"))
    return runs, skipped


def parse_notebook(source):
    """`source`: a path to an .ipynb file or an already loaded notebook dict.
    -> {"runs": [{"cell_index", "n_epochs", "epochs"}], "skipped": [str]}"""
    nb = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    out = {"runs": [], "skipped": []}
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        text = _output_texts(cell)
        if "poch" not in text:
            continue
        runs, skipped = parse_text(text)
        for r in runs:
            out["runs"].append({"cell_index": i, "n_epochs": len(r), "epochs": r})
        out["skipped"] += [f"cell {i}: {s}" for s in skipped]
    return out


NEUTRAL_GPU = 66.0       # not in a notebook; the middle of the range the models were trained with


def to_run(run_id, parsed_run, context=None, commits_per_day=2.0):
    """A parsed run -> the input dict of predict_live.predict_run(). Missing signals are assumed, and listed."""
    ctx = {"drift_severity_score": 0.0, "hyperparam_tuning_logged": 0, "commit_gap_days": 1.0,
           "notebook_edit_freq_per_day": 3.0, "drift_source": "default_0_no_score"}
    ctx.update(context or {})
    epochs = [{"epoch": r["epoch"], "train_accuracy": r["train_accuracy"], "val_accuracy": r["val_accuracy"],
               "loss": r["loss"], "gpu_utilization_pct": NEUTRAL_GPU, "commits_that_day": commits_per_day}
              for r in parsed_run["epochs"]]
    return {"run_id": run_id, "epochs": epochs, "context": ctx,
            "assumed": ["gpu_utilization_pct (not in a notebook)", "commits_that_day (repository average)"]}


def main():
    ap = argparse.ArgumentParser(description="Find training runs in a notebook.")
    ap.add_argument("notebook")
    ap.add_argument("--score", action="store_true", help="also score each run with Component 3")
    a = ap.parse_args()
    res = parse_notebook(a.notebook)
    print(f"{a.notebook}: {len(res['runs'])} usable training run(s), {len(res['skipped'])} skipped")
    for s in res["skipped"]:
        print("  skipped:", s)
    for n, r in enumerate(res["runs"], 1):
        e = r["epochs"]
        print(f"  run {n} (cell {r['cell_index']}): {r['n_epochs']} epochs, final train acc {e[-1]['train_accuracy']:.2f}, "
              f"val acc {e[-1]['val_accuracy']:.2f}, loss {e[-1]['loss']:.3f}")
        if a.score:
            from ..predict_live import predict_run
            out = predict_run(to_run(f"{Path(a.notebook).stem}_run{n}", r))
            print(f"     -> {out['severity_level']}: {out['title']}  (anomaly {out['anomaly_score']}, "
                  f"cause {out['recommended_cause']})")


if __name__ == "__main__":
    main()
