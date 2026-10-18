"""
Step 4a: Create a SIMULATED team chat for each repo (public repos have no chat).

The chat follows what really happened in the repo: on days a member worked on
something, they may post an update about that lifecycle stage, and teammates
may reply. Each member gets a fixed "talkativeness" so some people chat a lot
and some rarely - like a real team. Fixed seed = same chat every run.

Output: component2/data/chat/<team_id>.csv  (msg_id, timestamp, sender, message, reply_to)
This is labelled as SIMULATED data in the report.

Run from backend:
    python -m component2.chat.simulate_chat
"""
import csv
import random
from collections import defaultdict
from datetime import datetime, timedelta

from component2.config import CHAT_DIR
from component2.db import connect
from component2.research_teams import research_team_ids

SEED = 11

UPDATES = {
    "data_cleaning": ["Cleaned the dataset, removed duplicates and fixed missing values",
                      "Merged the two CSV files, some columns had wrong types",
                      "Handled the nulls, data is ready now"],
    "eda": ["Pushed some EDA plots, the distribution looks skewed",
            "Correlation heatmap is in the notebook, check it out",
            "Did summary stats, a few outliers we should discuss"],
    "feature_engineering": ["Added new features and encoded the categorical columns",
                            "Scaled the numeric features and did the train/test split",
                            "Tried feature selection, dropped a few weak columns"],
    "model_development": ["Trained the first model, results are in the notebook",
                          "Tuning hyperparameters now, will take a while",
                          "Pushed a new version of the model"],
    "model_evaluation": ["Evaluated the model on the test set, numbers look okay",
                         "Made the confusion matrix, one class is weak",
                         "Compared the models, sharing the results table"],
    "deployment_mlops": ["Set up the API so we can call the model",
                         "Added the Dockerfile and requirements",
                         "Saved the trained model and wrote a small pipeline"],
    "documentation": ["Updated the README with how to run it",
                      "Wrote the methodology part of the report",
                      "Added comments and markdown notes to the notebook"],
}
REPLIES = ["Nice, I'll take a look", "Great work!", "Can you explain that part tomorrow?",
           "I think we should double check that", "Okay, I'll continue from there",
           "Looks good to me", "Did you push it to main?", "Thanks, that helps my part"]
SOCIAL = ["Meeting tomorrow at 4?", "Who is free this weekend to work on it?",
          "Reminder: the submission is next week", "I'll be offline today"]


def stage_of_each_activity(conn):
    """Use the trained model's labels if they exist, otherwise the keyword baseline."""
    labels = {}
    for labeller in ("baseline", "model"):          # model overwrites baseline when available
        for aid, stage in conn.execute("SELECT activity_id, stage FROM stage_labels WHERE labeller = ?",
                                       (labeller,)).fetchall():
            labels[aid] = stage
    return labels


def simulate_team(rows, labels, rng):
    """rows: (activity_id, member_id, ts). Returns a list of chat messages."""
    members = sorted({m for _, m, _ in rows})
    talk = {m: rng.uniform(0.2, 0.9) for m in members}         # each person's talkativeness
    work = defaultdict(list)                                    # (day, member) -> stages that day
    for aid, member, ts in rows:
        stage = labels.get(aid, "other")
        if stage != "other":
            work[(ts[:10], member)].append(stage)

    msgs = []
    for (day, member), stages in sorted(work.items()):
        if rng.random() > talk[member]:
            continue
        main_stage = max(set(stages), key=stages.count)
        t = datetime.fromisoformat(day) + timedelta(hours=rng.randint(9, 21), minutes=rng.randint(0, 59))
        update = {"msg_id": f"m{len(msgs) + 1}", "timestamp": t, "sender": member,
                  "message": rng.choice(UPDATES[main_stage]), "reply_to": ""}
        msgs.append(update)
        for other in members:
            if other != member and rng.random() < talk[other] * 0.6:
                t = t + timedelta(minutes=rng.randint(2, 50))
                msgs.append({"msg_id": f"m{len(msgs) + 1}", "timestamp": t, "sender": other,
                             "message": rng.choice(REPLIES), "reply_to": update["msg_id"]})
        if rng.random() < 0.15:
            msgs.append({"msg_id": f"m{len(msgs) + 1}", "timestamp": t + timedelta(hours=1),
                         "sender": rng.choice(members), "message": rng.choice(SOCIAL), "reply_to": ""})
    msgs.sort(key=lambda m: m["timestamp"])
    return msgs


def main():
    rng = random.Random(SEED)
    conn = connect()
    labels = stage_of_each_activity(conn)
    teams = research_team_ids(conn)          # never add simulated chat to a real team
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    for team in teams:
        rows = conn.execute("SELECT activity_id, member_id, ts FROM activities "
                            "WHERE team_id = ? AND source != 'chat' ORDER BY ts", (team,)).fetchall()
        msgs = simulate_team(rows, labels, rng)
        with open(CHAT_DIR / f"{team}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["msg_id", "timestamp", "sender", "message", "reply_to"])
            w.writeheader()
            for m in msgs:
                w.writerow({**m, "timestamp": m["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ")})
        print(f"  {team:<45} {len(msgs):>4} simulated messages")
    print(f"Saved to {CHAT_DIR}")


if __name__ == "__main__":
    main()