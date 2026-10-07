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
