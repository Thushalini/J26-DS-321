# Coding Guide — DS Lifecycle Stages (for raters)

**Your job:** read one item (a commit message + files, a notebook cell, or a chat message) and pick the ONE stage it mostly belongs to. Work alone — don't discuss with the other rater until both are done.

**Rules**

1. Pick the stage of the *main* action. "Clean data and plot histogram" → the part with more code/effort wins; if equal, pick the earlier stage.
2. Use file names as clues (`Dockerfile` → deployment, `README.md` → documentation, `train.py` → model development).
3. Chat that is pure social talk / logistics ("ok", "meeting at 3?") → `other`.
4. If you truly can't tell → `other`. Don't guess.

|#|Stage (label to write)|It's this stage when the item is about…|Typical words / code|
|-|-|-|-|
|1|`data\_cleaning`|getting data in and fixing it: loading, merging, missing values, duplicates, wrong types, outliers removed|`read\_csv`, `dropna`, `fillna`, `merge`, "clean", "missing", "dtype", "scrape", "dataset"|
|2|`eda`|looking at the data to understand it: plots, summary stats, correlations, distributions|`describe()`, `histplot`, `corr()`, `value\_counts`, "distribution", "skewed", "explore"|
|3|`feature\_engineering`|making the model's inputs: new columns, encoding, scaling, feature selection, train/test split|`get\_dummies`, `StandardScaler`, `LabelEncoder`, `train\_test\_split`, "feature", "encode"|
|4|`model\_development`|building or training a model, choosing algorithms, tuning hyperparameters|`.fit(`, `RandomForest`, `XGBoost`, `GridSearchCV`, `epochs`, "train", "tune", "baseline model"|
|5|`model\_evaluation`|measuring how good a model is: metrics, validation, confusion matrix, comparing models|`f1\_score`, `accuracy`, `confusion\_matrix`, `cross\_val\_score`, "evaluate", "AUC", "overfit"|
|6|`deployment\_mlops`|putting the model to use or automating: APIs, Docker, CI/CD, pipelines, saving models, experiment tracking|`Dockerfile`, `FastAPI`, `pickle.dump`, `mlflow`, `.github/workflows`, "deploy", "endpoint"|
|7|`documentation`|explaining the work: README, reports, markdown notes, comments, docstrings, slides|`README.md`, markdown cells, "docs", "report", "update readme"|
|–|`other`|social talk, scheduling, merge commits with no content, unclear|"ok", "thanks", "meeting tomorrow?"|

## Examples (2–3 per stage)

**data\_cleaning**

* Commit: "Clean missing tenure values and fix TotalCharges dtype"
* Cell: `df = df.drop\_duplicates(); df\['age'] = df\['age'].fillna(df\['age'].median())`
* Chat: "Cleaning done, tenure had missing values"

**eda**

* Commit: "EDA: tenure distribution and summary stats"
* Cell: `sns.heatmap(df.corr(), annot=True)`
* Chat: "Tenure is right-skewed, see histogram"

**feature\_engineering**

* Commit: "Feature engineering: charges per month + one-hot encoding"
* Cell: `X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, stratify=y)`
* Chat: "Let's log-transform income before scaling"

**model\_development**

* Commit: "Train random forest baseline"
* Cell: `model = XGBClassifier(max\_depth=6).fit(X\_train, y\_train)`
* Chat: "Should we try XGBoost instead of random forest?"

**model\_evaluation**

* Commit: "Evaluate model with F1 on test set"
* Cell: `print(classification\_report(y\_test, y\_pred))`
* Chat: "F1 is 0.81 on test" / "Maybe accuracy is enough as metric?"

**deployment\_mlops**

* Commit: "Add Dockerfile and FastAPI serving endpoint"
* Cell: `joblib.dump(model, 'model.pkl')`
* Chat: "Dockerfile + API pushed"

**documentation**

* Commit: "Update README with results and setup steps"
* Cell (markdown): `## 3. Results — the random forest reached F1 = 0.81`
* Chat: "I'll write the methodology section of the report tonight"

## Tricky cases (agreed rules)

* Chat *asking* about a metric ("is accuracy enough?") → `model\_evaluation` (it's about how we judge the model).
* Saving a cleaned CSV → `data\_cleaning`. Saving a trained model → `deployment\_mlops`.
* `train\_test\_split` alone → `feature\_engineering` (preparing inputs), not model development.
* Markdown cell that is only a title ("# Churn prediction") → `documentation`.
* Requirements/env files (`requirements.txt`, `environment.yml`) → `deployment\_mlops`.
* Markdown prose or headings count as documentation.
* Installs, environment setup and MLflow count as deployment.
* Data ingestion counts as data cleaning.
* Imports-only cells count as other, unless they’re all plotting.

