"""Tests for Component 3's notebook log reader (finds training runs in .ipynb outputs)."""
from component3.collector import notebook_logs as N


def nb(*texts):
    return {"cells": [{"cell_type": "code", "outputs": [{"output_type": "stream", "text": t}]} for t in texts]
            + [{"cell_type": "markdown", "source": ["notes"]}]}


def keras2(n=4):
    return "".join(
        f"Epoch {e}/{n}\n 1/63 [....] - ETA: 5s - loss: {0.9 - e * 0.1:.4f} - accuracy: 0.5000\r"
        f"63/63 [=====] - 1s 8ms/step - loss: {0.8 - e * 0.1:.4f} - accuracy: {0.55 + e * 0.08:.4f} "
        f"- val_loss: {0.82 - e * 0.08:.4f} - val_accuracy: {0.5 + e * 0.09:.4f}\n" for e in range(1, n + 1))


def test_keras2_takes_the_last_value_of_each_epoch():
    run = N.parse_notebook(nb(keras2()))["runs"][0]
    assert run["n_epochs"] == 4
    last = run["epochs"][-1]
    assert (last["loss"], last["train_accuracy"], last["val_accuracy"]) == (0.4, 0.87, 0.86)


def test_keras3_colour_codes():
    text = "".join(f"Epoch {e}/3\n\x1b[1m63/63\x1b[0m \x1b[32m━━━\x1b[0m 8ms/step - accuracy: {0.6 + e * 0.1:.4f} "
                   f"- loss: {0.7 - e * 0.1:.4f} - val_accuracy: {0.55 + e * 0.1:.4f} - val_loss: {0.75 - e * 0.1:.4f}\n"
                   for e in range(1, 4))
    assert N.parse_notebook(nb(text))["runs"][0]["n_epochs"] == 3


def test_percent_accuracy_becomes_a_fraction():
    text = "".join(f"Epoch [{e}/3], Loss: {1 - e * 0.2:.4f}, Train Acc: {60 + e * 8:.2f}%, Val Acc: {55 + e * 9:.2f}%\n"
                   for e in range(1, 4))
    last = N.parse_notebook(nb(text))["runs"][0]["epochs"][-1]
    assert last["train_accuracy"] == 0.84 and last["val_accuracy"] == 0.82


def test_validation_numbers_are_not_mistaken_for_training_numbers():
    text = "".join(f"Epoch: {e:02d} | Train Loss: {0.9 - e * 0.1:.3f} | Train Acc: {0.6 + e * 0.05:.3f} | "
                   f"Val. Loss: {0.95 - e * 0.1:.3f} | Val. Acc: {0.55 + e * 0.06:.3f}\n" for e in range(1, 6))
    last = N.parse_notebook(nb(text))["runs"][0]["epochs"][-1]
    assert last["loss"] == 0.4 and last["train_accuracy"] == 0.85 and last["val_accuracy"] == 0.85


def test_several_prints_for_one_epoch_are_one_epoch():
    steps = "".join(f"Epoch [{e}/3], Step [{s}/3], Loss: {1.0 - e * 0.2:.4f}, Train Acc: {50 + e}%, Val Acc: {45 + e}%\n"
                    for e in range(1, 4) for s in (1, 2, 3))
    assert N.parse_notebook(nb(steps))["runs"][0]["n_epochs"] == 3


def test_a_cell_run_twice_gives_two_runs():
    assert len(N.parse_notebook(nb(keras2() + keras2()))["runs"]) == 2


def test_runs_without_validation_or_with_too_few_epochs_are_skipped_and_explained():
    no_val = "".join(f"Epoch {e}/5 - loss: {1 - e * 0.1:.3f} - accuracy: {0.5 + e * 0.05:.3f}\n" for e in range(1, 6))
    res = N.parse_notebook(nb(no_val))
    assert res["runs"] == [] and "val_accuracy" in res["skipped"][0]
    short = "Epoch 1/2 - loss: 0.5 - accuracy: 0.7 - val_loss: 0.6 - val_accuracy: 0.65\n" \
            "Epoch 2/2 - loss: 0.4 - accuracy: 0.8 - val_loss: 0.5 - val_accuracy: 0.7\n"
    assert N.parse_notebook(nb(short))["runs"] == []


def test_text_that_is_not_a_training_log_is_ignored():
    assert N.parse_notebook(nb("Number of epochs: 20\nbest score 0.9\n")) == {"runs": [], "skipped": []}


def test_to_run_lists_what_it_had_to_assume():
    run = N.to_run("r1", N.parse_notebook(nb(keras2()))["runs"][0])
    assert run["epochs"][0]["gpu_utilization_pct"] == N.NEUTRAL_GPU
    assert any("gpu" in a for a in run["assumed"])
