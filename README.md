# AI Bias Detector 🔍

A Python tool that trains an income prediction model on census-style data, then audits the model's predictions for demographic bias across sex and race.

Built as a personal project to explore responsible AI and algorithmic fairness — inspired by real cases like the ProPublica COMPAS investigation and Amazon's hiring algorithm that was found to penalise women's CVs.

---

## The Problem This Solves

AI models are increasingly used to make decisions that affect people's lives — hiring, lending, insurance, healthcare. A model can score highly on accuracy while still treating certain groups unfairly. This tool makes that visible.

> "A model that is 90% accurate overall can still be deeply unfair to specific groups. Accuracy hides what equity reveals."

---

## What It Does

**Step 1 — Load real data**
Loads the UCI Adult Income Dataset — real US Census data containing 30,162 records. Each record includes demographic information (age, sex, race, education, occupation, hours worked) and whether the person earns above or below $50K/year.

**Step 2 — Train a model**
Trains a Decision Tree classifier to predict income. The model never sees race or sex directly — it only uses education, occupation, hours, age and capital gains.

**Step 3 — Audit for bias**
Compares the model's positive prediction rates across demographic groups. Flags any group receiving significantly fewer high-income predictions than the overall average.

---

## Real Output

```
 AI BIAS DETECTOR
=========================================================
Loading real US Census data...

  Records:  30,162
  Source:   UCI Adult Income Dataset (real US Census data)
  Income >$50K: 7,508 (24.9%)

=========================================================
STEP 1: Training income prediction model
=========================================================

  Model:    Decision Tree Classifier (max_depth=6)
  Accuracy: 82.7%
  Test set: 9,049 records

  NOTE: High accuracy can hide unfair treatment of groups.
  This is why a bias audit is essential.

=========================================================
BIAS AUDIT: grouped by 'sex'
=========================================================

  Rate model predicts high income (>$50K):

  Male                            17.2%  ████████  (n=6,138)
  Female                           7.7%  ███  (n=2,911)

  OVERALL                         14.1%

  No significant bias flagged for 'sex'.

=========================================================
BIAS AUDIT: grouped by 'race'
=========================================================

  Rate model predicts high income (>$50K):

  White                           15.2%  ███████  (n=7,797)
  Asian-Pac-Islander              10.4%  █████  (n=259)
  Amer-Indian-Eskimo               9.2%  ████  (n=76)
  Black                            6.7%  ███  (n=861)
  Other                            3.6%  █  (n=56)

  OVERALL                         14.1%

  WARNING: 'Other'
  Predicted high-income rate is 10.5% below average.
  The model is significantly less likely to predict high
  income for this group — a potential fairness concern.

=========================================================
Audit complete.
A model can score well on accuracy while still being unfair.
Responsible AI means auditing both.
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/amal-mahmood7/AI-Bias-Detector-.git
cd AI-Bias-Detector-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the audit
```bash
python bias_detector.py
```

---

## Requirements

- Python 3.7+
- pandas
- scikit-learn

---

## Why This Matters in the Real World

| Case | What happened |
|---|---|
| Amazon hiring algorithm (2018) | Penalised CVs containing the word "women's" |
| COMPAS recidivism tool | Falsely flagged Black defendants as high risk at twice the rate of white defendants |
| UK A-Level algorithm (2020) | Downgraded students from lower-income schools using historical data |
| Healthcare allocation (2019) | Recommended less care for Black patients than equally sick white patients |

In each case, the model was not explicitly programmed to discriminate. Bias emerged from the data.

---

## Key Concepts Demonstrated

- **Proxy discrimination** — how models learn bias indirectly through correlated features
- **Accuracy vs fairness trade-off** — why a high accuracy score is not enough
- **Demographic parity** — the principle that positive prediction rates should be roughly equal across groups
- **Disparate impact** — when a neutral-seeming policy produces unequal outcomes

---

## Responsible AI Recommendations

1. **Investigate features** — which inputs correlate with protected characteristics?
2. **Apply fairness constraints** — retrain with fairness-aware algorithms
3. **Rebalance data** — historical data reflects historical inequality
4. **Audit continuously** — bias can reappear after model updates
5. **Consult affected communities** — technical metrics are not a substitute for human judgement

Aligned with: **EU AI Act** | **IEEE Ethically Aligned Design** | **Microsoft Fairlearn**

---

## About

Built by **Amal Mahmood**, BSc Artificial Intelligence student at Oxford Brookes University.

Personal project exploring responsible AI and algorithmic fairness.
