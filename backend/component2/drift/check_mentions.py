from component2.drift.mentions import find_mentions

examples = [
    "y = df['SalePrice']; LinearRegression().fit; rmse and r2_score; KNeighborsClassifier",
    "Train RF baseline, print(f1_score(y_test, pred))",
    "switch to XGBClassifier, accuracy is 0.91",
    "target variable is Churn",
]

for text in examples:
    print(text)
    print("   ->", {dim: dict(c) for dim, c in find_mentions(text).items()})