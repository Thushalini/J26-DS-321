"""Tests for the run reporter (student side) and for how the server completes a reported run."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from component3.api import service
from component3.client import c3_reporter as R


def _epochs(rep, n=5):
    for e in range(1, n + 1):
        rep.epoch(loss=1.0 - e * 0.1, train_accuracy=0.5 + e * 0.05, val_accuracy=0.45 + e * 0.05, val_loss=1.1 - e * 0.1)


def test_payload_holds_the_numbers_of_every_epoch():
    rep = R.RunReporter("http://x", "team1", "key", "demo")
    _epochs(rep)
    p = rep.payload()
    assert p["run_id"].startswith("demo_") and len(p["epochs"]) == 5
    first = p["epochs"][0]
    assert first["epoch"] == 1 and abs(first["train_accuracy"] - 0.55) < 1e-9 and abs(first["loss"] - 0.9) < 1e-9


def test_fewer_than_three_epochs_are_not_sent():
    rep = R.RunReporter("http://127.0.0.1:1", "t", "k")
    _epochs(rep, 2)
    assert rep.finish() is None


def test_a_server_that_cannot_be_reached_does_not_raise():
    rep = R.RunReporter("http://127.0.0.1:1", "t", "k", timeout=1)
    _epochs(rep)
    assert rep.finish() is None


def test_run_is_posted_with_the_key_header():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["key"] = self.path, self.headers.get("X-C3-Key")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"severity_level": "LOW", "title": "No issue detected"}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rep = R.RunReporter(f"http://127.0.0.1:{srv.server_port}", "team1", "secret", "demo")
    _epochs(rep)
    out = rep.finish()
    srv.shutdown()
    assert out["severity_level"] == "LOW"
    assert seen["path"] == "/c3/teams/team1/runs" and seen["key"] == "secret" and len(seen["body"]["epochs"]) == 5


def test_keras_callback_reports_epochs_and_skips_epochs_without_validation():
    cb = R.C3Keras("http://127.0.0.1:1", "t", "k", "keras_demo")
    cb.on_epoch_end(0, {"loss": 0.9, "accuracy": 0.6, "val_accuracy": 0.55, "val_loss": 1.0})
    cb.on_epoch_end(1, {"loss": 0.8, "accuracy": 0.7})                      # no validation numbers
    assert len(cb.reporter.epochs) == 1


def test_server_fills_in_what_the_reporter_cannot_know(monkeypatch):
    from component3.collector import drift_reader
    monkeypatch.setattr(drift_reader, "read_drift", lambda team_id: {"drift_severity_score": 0.7, "source": "component2"})
    rep = R.RunReporter("http://x", "t", "k")
    _epochs(rep)
    out = service.complete_run("t", rep.payload())
    assert all(e["gpu_utilization_pct"] == service.NEUTRAL_GPU for e in out["epochs"])
    assert out["context"]["drift_severity_score"] == 0.7 and out["context"]["drift_source"] == "component2"
    assert "gpu_utilization_pct (not measured)" in out["assumed"]


def test_no_drift_score_gives_zero_and_says_so(monkeypatch):
    from component3.collector import drift_reader
    monkeypatch.setattr(drift_reader, "read_drift", lambda team_id: {"drift_severity_score": None, "source": "no_score"})
    rep = R.RunReporter("http://x", "t", "k")
    _epochs(rep)
    out = service.complete_run("t", rep.payload())
    assert out["context"]["drift_severity_score"] == 0.0 and out["context"]["drift_source"] == "default_0_no_score"
