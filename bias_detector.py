"""
AI Bias Detector
----------------
Trains an income prediction model on real US Census data,
then audits the model's predictions for demographic bias
across sex and race.

This mirrors real industry practice used by AI ethics teams
to check whether models treat different groups fairly.

Dataset: UCI Adult Income Dataset (real US Census data, 48,842 records)
Source:  https://archive.ics.uci.edu/dataset/2/adult

Inspired by: ProPublica COMPAS investigation and Amazon's hiring
algorithm that was found to penalise women's CVs.

Author: Amal Mahmood
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def load_data(filepath="adult.data"):
    """
    Load the UCI Adult Income Dataset.
    Real US Census data containing demographic and employment
    information for 48,842 individuals.
    """
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]

    df = pd.read_csv(
        filepath,
        names=columns,
        sep=", ",
        engine="python",
        na_values="?"
    )
    df.dropna(inplace=True)
    df["income"] = df["income"].str.strip()

    print(f"  Records:  {len(df):,}")
    print(f"  Source:   UCI Adult Income Dataset (real US Census data)")
    print(f"  Income >$50K: {(df['income'] == '>50K').sum():,} ({(df['income'] == '>50K').mean()*100:.1f}%)\n")
    return df


def train_model(df):
    """
    Train a Decision Tree classifier to predict income.
    The model uses education, occupation, age, hours and capital gains.
    It never sees race or sex directly.
    """
    print("=" * 57)
    print("STEP 1: Training income prediction model")
    print("=" * 57)

    audit_cols = df[["sex", "race"]].copy()
    df_model = df.copy()

    le = LabelEncoder()
    for col in ["workclass", "education", "marital_status",
                "occupation", "relationship", "native_country", "income"]:
        df_model[col] = le.fit_transform(df_model[col].astype(str))

    features = [
        "age", "education_num", "hours_per_week",
        "capital_gain", "capital_loss", "occupation", "workclass"
    ]

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
    print(f"  This is why a bias audit is essential.\n")

    return preds, y_test, aud_test


def audit_bias(preds, y_test, audit_df, group_col):
    """
    Compare the model's positive prediction rates across demographic groups.
    Flags any group receiving significantly fewer high-income predictions.
    """
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
        print(f"  {group:<30} {info['rate']:5.1f}%  {bar}  (n={info['n']:,})")

    print(f"\n  {'OVERALL':<30} {overall:5.1f}%\n")
    print("─" * 57)
    print("  BIAS FLAGS (groups 10%+ below overall rate):\n")

    flagged = False
    for group, info in sorted(group_rates.items(), key=lambda x: x[1]["rate"]):
        gap = overall - info["rate"]
        if gap > 10:
            print(f"  WARNING: '{group}'")
            print(f"  Predicted high-income rate is {gap:.1f}% below average.")
            print(f"  The model is significantly less likely to predict high")
            print(f"  income for this group — a potential fairness concern.\n")
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
     inequality. Using them can encode discrimination into the
     model even without race or sex being explicit inputs.
     This is called proxy discrimination.

  2. APPLY FAIRNESS CONSTRAINTS
     Retrain with fairness-aware algorithms that penalise
     disparate outcomes across protected groups.
     Tools like Microsoft Fairlearn can help with this.

  3. REBALANCE TRAINING DATA
     Historical data reflects historical bias. Reweighting
     or resampling can reduce inherited disparities.

  4. AUDIT CONTINUOUSLY
     Bias can reappear when data or model changes. Build
     auditing into the deployment pipeline, not just build.

  5. CONSULT AFFECTED COMMUNITIES
     No fairness metric replaces asking the people affected
     whether the system treats them fairly.

  Frameworks: EU AI Act | IEEE Ethically Aligned Design | Fairlearn
    """)


if __name__ == "__main__":
    print("\n AI BIAS DETECTOR")
    print("=" * 57)
    print("Loading real US Census data (UCI Adult Income Dataset)...\n")

    df = load_data("adult.data")
    preds, y_test, audit_df = train_model(df)

    audit_bias(preds, y_test, audit_df, group_col="sex")
    audit_bias(preds, y_test, audit_df, group_col="race")

    recommendations()

    print("=" * 57)
    print("Audit complete.")
    print("A model can score well on accuracy while still being unfair.")
    print("Responsible AI means auditing both.\n")
