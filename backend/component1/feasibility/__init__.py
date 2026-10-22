"""The feasibility classifier (Module 3's trained model): is this plan feasible, risky or infeasible?

    simulate.py   simulated project plans, labelled by what WOULD happen to them (not by plan rules)
    train.py      train and compare with the fixed-rule baseline; save the model and results
    predict.py    judge one blueprint: label, probability, reasons (SHAP) and changes that would help

READ BEFORE QUOTING ANY NUMBER: the training data is simulated. A score shows what the model
learned from the simulation. The expert-labelled sheet (expert_sheet.py) is the real-world check.
"""
