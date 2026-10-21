from component4.m4_readiness.rubric import assess

FILES = ["README.md", "requirements.txt", "src/train.py", "tests/test_train.py", ".github/workflows/ci.yml",
         "Dockerfile", "dvc.yaml", "data/raw.csv.dvc", "monitoring/drift_report.py", "docs/model_card.md"]


def test_no_repo_means_no_score():
    assert assess([]) is None


def test_scores_are_between_zero_and_one_hundred_per_area():
    result = assess(FILES)
    assert 0 <= result["score"] <= 100
    assert {a["id"] for a in result["areas"]} == {"versioning", "monitoring", "pipeline", "documentation"}
    assert all(0 <= a["score"] <= 100 for a in result["areas"])


def test_more_evidence_means_a_higher_score():
    assert assess(FILES)["score"] > assess(["README.md", "train.py"])["score"]


def test_each_found_criterion_names_the_file_that_proved_it():
    pipeline = next(a for a in assess(FILES)["areas"] if a["id"] == "pipeline")
    assert {item["file"] for item in pipeline["found"]} >= {".github/workflows/ci.yml", "Dockerfile"}
    assert pipeline["score"] == 100 and not pipeline["missing"]


def test_missing_criteria_are_listed_in_the_reason():
    versioning = next(a for a in assess(["README.md"])["areas"] if a["id"] == "versioning")
    assert versioning["score"] == 0 and "dependencies listed" in versioning["reason"]
