# Buddy.ds — backend

Python backend. The web framework (FastAPI or Django) is **not decided yet**, so this folder only holds the structure.

## Structure

```
component1/        Intelligent DS Project Planning & Decision Facilitation
component2/        DS Behavioural Modelling & Contribution Analytics
component3/        Real-Time Collaboration Intelligence & Adaptive Intervention
component4/        Predictive Success Evaluation & Continuous MLOps Readiness
tests/             Tests
requirements.txt   Python packages (empty until the framework is chosen)
```

Each member works inside their own component folder.

## Getting started

You need Python 3.11 or newer. Run these from the `backend/` folder.

Create a virtual environment (once):

```bash
python -m venv .venv
```

Activate it (every new terminal) — Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

When you add a package, add it to `requirements.txt` in the same commit so everyone gets it.

## Running the Buddy.ds app (demo for teams and supervisors)

The app layer lives in `buddy/` (accounts, teams, chat, blueprint, feedback, live sync). It only **adds**
tables (`users`, `sessions`, `teams`, `team_members`, `chat_messages`, `blueprints`, `feedback`) and never
changes the Phase 1 research rows. The research scripts in `component2/` only process research teams
(`component2/research_teams.py`), so re-running them can't touch real teams.

1. In `backend/.env` set `STAFF_CODE=<any secret>`. Only **examiner** accounts need it (they read all feedback);
   give it to your examiners yourself. Supervisors register freely and get access to a team through its join code.
2. Back up the database first, then register the research teams and the demo staff accounts (safe to re-run):
   ```bash
   python -m buddy.seed_demo --password "<demo password>"
   ```
   This creates `supervisor@demo.buddy` and `examiner@demo.buddy`.
3. Start the API: `uvicorn main:app --reload`. Then start the frontend: `cd ../frontend && npm run dev`.
4. Open http://localhost:3000/guide for each stakeholder's flow.

| Who | Sees | Can do |
| --- | --- | --- |
| Student (lead) | own teams | create team + repo link, blueprint, chat, sync, link git identities |
| Student (member) | own teams | join with code, chat, blueprint, sync, link git identities |
| Supervisor | supervised teams + research dataset (read-only) | join with code, chat, sync, read feedback (students anonymous) |
| Examiner | research dataset (read-only) | read all feedback, download CSV |

A sync clones the repo into a temporary folder, runs the same Component 2 pipeline (keyword baseline +
trained model, drift against the team's blueprint, weekly roles) and takes 1-2 minutes. Private repos need
`GITHUB_TOKEN` to have read access. The `/c2/...` endpoints stay open for C3 / C4; the frontend uses the
access-checked `/app/teams/{id}/behavior/...` copies.
