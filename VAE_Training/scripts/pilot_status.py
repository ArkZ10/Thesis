"""Read-only pilot monitor; standard library only, no GPU or model access."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def percent(value):
    return f"{float(value):.2%}" if value not in (None, "") else "-"


def number(value):
    return f"{float(value):.4f}" if value not in (None, "") else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path)
    args = parser.parse_args()
    if args.directory:
        directory = args.directory.resolve()
    else:
        launch = ROOT / "manifests/pilot_launch.json"
        if not launch.exists():
            print("Pilot not launched yet.")
            return
        directory = Path(json.loads(launch.read_text())["directory"])
    status_path = directory / "status.json"
    if not status_path.exists():
        print("Waiting for pilot process:", directory)
        return
    state = json.loads(status_path.read_text())
    print("Run:", directory.name)
    print("Stage:", state.get("stage"), "| PID:", state.get("pid"))
    print(f"Epoch {state.get('epoch', 0)}/{state.get('epochs', 3)} | batch {state.get('batch', 0)}/{state.get('batches', 0)} | {state.get('percent', 0):.1f}%")
    print(f"Running train loss: {number(state.get('train_loss'))} | token accuracy: {percent(state.get('train_token_accuracy'))}")
    eta = state.get("epoch_training_eta_seconds")
    if state.get("stage") == "training" and eta is not None:
        print(f"Epoch training ETA: {eta / 60:.1f} minutes (validation/checkpoint time excluded)")
    if state.get("evaluation_total", 0):
        print(f"Evaluation: {state.get('evaluation_completed', 0)}/{state['evaluation_total']} sentences | exact so far: {state.get('evaluation_exact', '-')}")
    if state.get("error"):
        print("ERROR:", state["error"])
    print("\nPer-epoch metrics (token accuracy is teacher-forced; exact is free reconstruction):")
    print(f"{'Epoch':>5} {'Train loss':>11} {'Train tok':>10} {'Val loss':>10} {'Val tok':>10} {'Val exact':>11} {'Numeric':>10}")
    metrics = directory / "metrics.csv"
    if metrics.exists():
        with metrics.open() as stream:
            for row in csv.DictReader(stream):
                epoch = "base" if row["epoch"] == "0" else row["epoch"]
                print(f"{epoch:>5} {number(row['train_loss']):>11} {percent(row['train_token_accuracy']):>10} {number(row['val_loss']):>10} {percent(row['val_token_accuracy']):>10} {percent(row['val_exact_match_rate']):>11} {percent(row['val_numeric_match_rate']):>10}")
    print("\nCtrl+C exits this monitor; it does not stop training.")


if __name__ == "__main__":
    main()
