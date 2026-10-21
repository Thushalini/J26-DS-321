"""c3_reporter: report a training run to Component 3 from ANY training loop. One file, no installs.

Copy this file next to your notebook (or add it to the project), then:

  Keras / TensorFlow
      from c3_reporter import C3Keras
      model.fit(X, y, validation_split=0.15, epochs=15,
                callbacks=[C3Keras("http://127.0.0.1:8000", TEAM_ID, API_KEY, "mlp_small")])

  PyTorch, scikit-learn or any loop
      from c3_reporter import RunReporter
      rep = RunReporter("http://127.0.0.1:8000", TEAM_ID, API_KEY, "my_run")
      for epoch in range(20):
          ...train...
          rep.epoch(loss=train_loss, train_accuracy=train_acc, val_accuracy=val_acc)
      rep.finish()               # sends the run; the alert appears on the team's Alerts page

What is sent: the numbers of each epoch (loss, training accuracy, validation accuracy, and GPU use if it
can be measured). No code and no data. Anything not measured (GPU use, commits, ...) is filled in by the
server with a neutral value and marked as assumed. Training never stops because of a reporting problem.
"""
import json
import time
import urllib.error
import urllib.request

try:                                            # Keras is optional: only needed for C3Keras
    from tensorflow import keras as _keras
    _Base = _keras.callbacks.Callback
except Exception:                               # no TensorFlow here: a plain base class keeps the file importable
    _Base = object


def _gpu_percent():
    """GPU utilisation in percent if an NVIDIA GPU and pynvml are available, else None (not measured)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return float(pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(0)).gpu)
    except Exception:
        return None


class RunReporter:
    def __init__(self, base_url, team_id, api_key, run_name="run", timeout=15):
        self.url = f"{base_url.rstrip('/')}/c3/teams/{team_id}/runs"
        self.api_key, self.timeout = api_key, timeout
        self.run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.epochs = []

    def epoch(self, loss, train_accuracy, val_accuracy, val_loss=None, gpu_utilization_pct=None):
        """Record one finished epoch. Accuracies are fractions between 0 and 1."""
        row = {"epoch": len(self.epochs) + 1, "loss": float(loss),
               "train_accuracy": float(train_accuracy), "val_accuracy": float(val_accuracy)}
        gpu = gpu_utilization_pct if gpu_utilization_pct is not None else _gpu_percent()
        if gpu is not None:
            row["gpu_utilization_pct"] = float(gpu)
        if val_loss is not None:
            row["val_loss"] = float(val_loss)
        self.epochs.append(row)

    def payload(self):
        return {"run_id": self.run_id, "epochs": self.epochs}

    def finish(self):
        """Send the run. Returns the server's answer (a dict), or None if it could not be sent."""
        if len(self.epochs) < 3:
            print(f"[Component 3] {len(self.epochs)} epochs recorded; at least 3 are needed, so nothing was sent.")
            return None
        req = urllib.request.Request(self.url, data=json.dumps(self.payload()).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "X-C3-Key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", "")
            except Exception:
                detail = ""
            print(f"[Component 3] the server refused the run ({e.code}): {detail}")
            return None
        except Exception as e:
            print(f"[Component 3] could not reach the server: {e}")
            return None
        print(f"[Component 3] {out.get('severity_level')}: {out.get('title')}")
        return out


class C3Keras(_Base):
    """Keras callback: reports every epoch, sends the run when training ends."""

    def __init__(self, base_url, team_id, api_key, run_name="keras_run"):
        if _Base is not object:
            super().__init__()
        self.reporter = RunReporter(base_url, team_id, api_key, run_name)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc, val_acc = logs.get("accuracy", logs.get("acc")), logs.get("val_accuracy", logs.get("val_acc"))
        if logs.get("loss") is None or acc is None or val_acc is None:
            return                                              # no validation data: nothing usable this epoch
        self.reporter.epoch(logs["loss"], acc, val_acc, logs.get("val_loss"))

    def on_train_end(self, logs=None):
        self.reporter.finish()
