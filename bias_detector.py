"""
AI Bias Detector
----------------
Trains a simple income prediction model on census-style data,
then audits the model's predictions for demographic bias.

This mirrors real industry practice - companies like Google and IBM
have dedicated teams that audit AI systems for unfair treatment
of protected groups before and after deployment.

Inspired by: UCI Adult Income Dataset findings and the ProPublica
COMPAS investigation into racial bias in criminal justice AI.

Author: Amal Mahmood
"""

import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def generate_data(n=3000, seed=42):
    """
    Generate census-style income data where demographic groups
    have different feature distributions - reflecting real labour
    market inequalities documented in US Census research.

    The model learns from these features, and the audit then checks
    whether its predictions treat groups differently.
    """
    random.seed(seed)
    rows = []

    group_configs = {
        "White_Male":             {"n": 900, "edu_mean": 11, "hours_mean": 45, "mgmt_prob": 0.35},
        "White_Female":           {"n": 600, "edu_mean": 10, "hours_mean": 38, "mgmt_prob": 0.15},
        "Asian-Pac-Islander_Male":{"n": 200, "edu_mean": 12, "hours_mean": 46, "mgmt_prob": 0.30},
        "Asian-Pac-Islander_Female":{"n":150,"edu_mean": 11, "hours_mean": 40, "mgmt_prob": 0.18},
        "Black_Male":             {"n": 300, "edu_mean": 9,  "hours_mean": 40, "mgmt_prob": 0.10},
        "Black_Female":           {"n": 250, "edu_mean": 9,  "hours_mean": 38, "mgmt_prob": 0.08},
        "Amer-Indian-Eskimo_Male":{"n": 150, "edu_mean": 8,  "hours_mean": 40, "mgmt_prob": 0.08},
        "Amer-Indian-Eskimo_Female":{"n":100,"edu_mean": 8,  "hours_mean": 36, "mgmt_prob": 0.05},
        "Other_Male":             {"n": 150, "edu_mean": 9,  "hours_mean": 42, "mgmt_prob": 0.10},
        "Other_Female":           {"n": 100, "edu_mean": 9,  "hours_mean": 38, "mgmt_prob": 0.08},
    }

    for key, cfg in group_configs.items():
        race, sex = key.rsplit("_", 1)
        for _ in range(cfg["n"]):
            edu = max(5, min(16, int(random.gauss(cfg["edu_mean"], 2))))
            hours = max(20, min(80, int(random.gauss(cfg["hours_mean"], 8))))
            is_mgmt = random.random() < cfg["mgmt_prob"]
            occupation = "Exec-managerial" if is_mgmt else random.choice([
                "Prof-specialty", "Adm-clerical", "Sales",
                "Other-service", "Craft-repair", "Transport-moving"
            ])
            capital_gain = random.choice([0] * 8 + [random.randint(2000, 15000)])
            age = random.randint(22, 62)

            # Income based purely on observable features
            # (bias emerges because groups have different feature distributions)
            score = (
                (edu - 5) * 0.08 +
                (hours - 20) * 0.006 +
                (0.35 if occupation == "Exec-managerial" else 0) +
                (0.15 if occupation == "Prof-specialty" else 0) +
                capital_gain * 0.00002 +
                random.gauss(0, 0.05)
            )
            income = ">50K" if score > 0.55 else "<=50K"

            rows.append({
                "age": age,
                "education_num": edu,
                "hours_per_week": hours,
                "capital_gain": capital_gain,
                "occupation": occupation,
                "sex": sex,
                "race": race,
                "income": income
            })

    df = pd.DataFrame(rows)
    random.shuffle(rows)
    df = pd.DataFrame(rows)
    print(f"  Records: {len(df):,}")
    print(f"  Source:  Based on UCI Adult Income Dataset demographic distributions\n")
    return df


def train_model(df):
    """Train a Decision Tree classifier to predict income."""
    print("=" * 57)
    print("STEP 1: Training income prediction model")
    print("=" * 57)

    audit_cols = df[["sex", "race"]].copy()
    df_model = df.copy()
    le = LabelEncoder()
    df_model["occupation"] = le.fit_transform(df_model["occupation"])
    df_model["income"] = le.fit_transform(df_model["income"])

    features = ["age", "education_num", "hours_per_week", "capital_gain", "occupation"]
    X = df_model[features]
    y = df_model["income"]

    X_train, X_test, y_train, y_test, aud_train, aud_test = train_test_split(
        X, y, audit_cols, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds) * 100
    print(f"\n  Model:    Decision Tree Classifier (max_depth=6)")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  Test set: {len(X_test):,} records")
    print(f"\n  NOTE: High accuracy can hide unfair treatment of groups.")
    print(f"  A bias audit looks beyond the headline number.\n")

    return preds, y_test, aud_test


def audit_bias(preds, y_test, audit_df, group_col):
    """Audit model predictions for disparities across demographic groups."""
    print("=" * 57)
    print(f"BIAS AUDIT: grouped by '{group_col}'")
    print("=" * 57)

    results = audit_df.copy()
    results["predicted_high_income"] = preds
    overall = results["predicted_high_income"].mean() * 100

    print(f"\n  Rate model predicts high income (>$50K):\n")

    group_rates = {}
    for group, subset in results.groupby(group_col):
        rate = subset["predicted_high_income"].mean() * 100
        group_rates[group] = {"rate": rate, "n": len(subset)}

    for group, info in sorted(group_rates.items(), key=lambda x: -x[1]["rate"]):
        bar = "█" * int(info["rate"] / 2)
        print(f"  {group:<28} {info['rate']:5.1f}%  {bar}  (n={info['n']:,})")

    print(f"\n  {'OVERALL':<28} {overall:5.1f}%\n")
    print("─" * 57)
    print("  BIAS FLAGS (groups 10%+ below overall):\n")

    flagged = False
    for group, info in sorted(group_rates.items(), key=lambda x: x[1]["rate"]):
        gap = overall - info["rate"]
        if gap > 10:
            print(f"  WARNING: '{group}'")
            print(f"  Predicted high-income rate is {gap:.1f}% below average.")
            print(f"  The model is significantly less likely to predict high income")
            print(f"  for this group — a potential fairness concern.\n")
            flagged = True

    if not flagged:
        print(f"  No significant bias flagged for '{group_col}'.\n")


def recommendations():
    print("\n" + "=" * 57)
    print("RESPONSIBLE AI RECOMMENDATIONS")
    print("=" * 57)
    print("""
  1. INVESTIGATE ROOT CAUSE
     Features like occupation and education reflect historical
     inequality. Using them can encode discrimination into the model
     even without any explicit protected attributes being used.

  2. APPLY FAIRNESS CONSTRAINTS
     Retrain with fairness-aware algorithms that penalise
     disparate outcomes across protected groups.

  3. REBALANCE TRAINING DATA
     Historical data reflects historical bias. Reweighting or
     resampling can reduce inherited disparities.

  4. AUDIT CONTINUOUSLY
     Bias can reappear when data or model changes. Build auditing
     into the deployment pipeline, not just the build stage.

  5. CONSULT AFFECTED COMMUNITIES
     No fairness metric replaces asking the people affected
     whether the system treats them fairly.

  Frameworks: EU AI Act | IEEE Ethically Aligned Design | Fairlearn
    """)


if __name__ == "__main__":
    print("\n AI BIAS DETECTOR")
    print("=" * 57)
    print("Generating census-style dataset...\n")

    df = generate_data(n=3000)
    preds, y_test, audit_df = train_model(df)

    audit_bias(preds, y_test, audit_df, group_col="sex")
    audit_bias(preds, y_test, audit_df, group_col="race")

    recommendations()

    print("=" * 57)
    print("Audit complete.")
    print("A model can score well on accuracy while still being unfair.")
    print("Responsible AI means auditing both.\n")
