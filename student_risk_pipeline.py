"""
Student Academic Risk Pipeline
==============================
Usage:  python student_risk_pipeline.py path/to/students.csv

Steps: 1 Exploration -> 2 Preprocessing -> 3 Risk definition + models
       -> 4 Evaluation -> 5 Prediction/Recommendation -> 6 Automated report
Outputs (in ./output): plots/*.png, risk_report.csv, console log.
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")                       # save plots to files, no GUI needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT = "output"; PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)
sns.set_theme(style="whitegrid")

# Everything printed to the console is ALSO saved to output/results.txt, so the
# findings and model metrics are preserved without having to re-run the script.
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for st in self.streams: st.write(text)
    def flush(self):
        for st in self.streams: st.flush()
_results_file = open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _results_file)

# =====================================================================
# 0. LOAD
# =====================================================================
path = sys.argv[1] if len(sys.argv) > 1 else "students.csv"
df = pd.read_csv(path)
df.columns = df.columns.str.strip()                 # guard against stray spaces in headers
ATT, TOTAL = "Attendance (%)", "Total_Score"
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

# =====================================================================
# 1. PREPROCESSING / CLEANING  (done first so the EDA is trustworthy)
# =====================================================================
# Why first: an attendance of 140% or a score of -10 would distort every
# correlation and graph, so we fix data quality before exploring.

# 1a. Duplicates: a Student_ID should appear once.
before = len(df)
df = df.drop_duplicates().drop_duplicates(subset="Student_ID", keep="first")
print(f"Removed {before - len(df)} duplicate rows")

# 1b. Force numeric columns to numeric (bad strings like 'N/A' become NaN).
score_cols = ["Midterm_Score", "Final_Score", "Assignments_Avg", "Quizzes_Avg",
              "Participation_Score", "Projects_Score", TOTAL,
              "math_score", "reading_score", "writing_score", "science_score"]
for c in score_cols + [ATT, "Age"]:
    # .str.strip() first: this dataset has stray tabs such as '\t41' in math_score
    df[c] = pd.to_numeric(df[c].astype("string").str.strip(), errors="coerce")

# 1c. Impossible values -> NaN (then imputed). Percentages/scores live in 0-100.
for c in score_cols + [ATT]:
    hi = 10 if c == "Participation_Score" else 100        # participation is scored out of 10
    bad = ((df[c] < 0) | (df[c] > hi)).sum()
    if bad: print(f"  {c}: {bad} out-of-range values set to missing")
    df.loc[(df[c] < 0) | (df[c] > hi), c] = np.nan
bad_age = ((df["Age"] < 15) | (df["Age"] > 60)).sum()
if bad_age: print(f"  Age: {bad_age} implausible values set to missing")
df.loc[(df["Age"] < 15) | (df["Age"] > 60), "Age"] = np.nan

# 1d. Tidy categoricals (case/whitespace inconsistencies).
for c in ["Department", "Gender", "test_preparation_course", "Grade"]:
    df[c] = df[c].astype("string").str.strip().str.title()
# Same department spelled several ways ('CS' / 'Computer Science', 'Math' / 'Mathematics').
# Left alone, the model would treat them as different departments.
df["Department"] = df["Department"].replace({"Cs": "Computer Science", "Math": "Mathematics"})
df["Grade"] = df["Grade"].str.upper()
# test_preparation_course is already 0/1 in this file; keep it as a clean text category.
df["test_preparation_course"] = df["test_preparation_course"].replace({"1": "Completed", "0": "Not Completed"})

print("\nMissing values per column (after fixing bad values):")
print(df.isna().sum()[lambda s: s > 0].to_string() or "none")

# 1e. Imputation. Median for numbers (robust to outliers), mode for categories.
# Imputed here because the risk LABEL needs Attendance and Total_Score for every
# student; the model pipeline later has its own imputer as a second safety net.
num_cols = df.select_dtypes("number").columns
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in ["Department", "Gender", "test_preparation_course"]:
    df[c] = df[c].fillna(df[c].mode()[0])

# =====================================================================
# 2. EXPLORATION
# =====================================================================
print("\n=== EDA ===")
print(df[[ATT, TOTAL, "Midterm_Score", "Final_Score"]].describe().round(1).to_string())

# 2a. Attendance vs marks: correlation (Pearson = linear, Spearman = rank-based)
pear = df[ATT].corr(df[TOTAL]); spear = df[ATT].corr(df[TOTAL], method="spearman")
print(f"\nAttendance vs Total_Score: Pearson r={pear:.3f}, Spearman rho={spear:.3f}")

plt.figure(figsize=(7, 5))
sns.regplot(data=df, x=ATT, y=TOTAL, scatter_kws={"alpha": .3, "s": 14}, line_kws={"color": "red"})
plt.title(f"Attendance vs Total Score (r = {pear:.2f})"); plt.tight_layout()
plt.savefig(f"{PLOTS}/01_attendance_vs_score.png", dpi=130); plt.close()

# 2b. Banded view: easier to communicate than a scatter cloud.
df["Attendance_Band"] = pd.cut(df[ATT], [0, 60, 75, 90, 100],
                               labels=["<60", "60-75", "75-90", "90+"], include_lowest=True)
band = df.groupby("Attendance_Band", observed=True)[TOTAL].agg(["mean", "count"]).round(1)
print("\nMean Total_Score by attendance band:\n", band.to_string())
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Attendance_Band", y=TOTAL, palette="Blues")
plt.title("Total Score by Attendance Band"); plt.tight_layout()
plt.savefig(f"{PLOTS}/02_score_by_attendance_band.png", dpi=130); plt.close()

# 2b2. Attendance vs GRADE. Total_Score may not follow attendance, so check the
# other outcome column too. Grade is mapped to 0-4 so we can correlate it.
grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
df["Grade_Num"] = df["Grade"].map(grade_map)
g_corr = df[ATT].corr(df["Grade_Num"], method="spearman")
print(f"Attendance vs Grade (A=4..F=0): Spearman rho={g_corr:.3f}")
print("Mean attendance by Grade:\n", df.groupby("Grade")[ATT].mean().round(1).reindex(list("ABCDF")).to_string())
print("Mean Total_Score by Grade:\n", df.groupby("Grade")[TOTAL].mean().round(1).reindex(list("ABCDF")).to_string())
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Grade", y=ATT, order=list("ABCDF"), palette="viridis")
plt.title(f"Attendance by Grade (Spearman rho = {g_corr:.2f})"); plt.tight_layout()
plt.savefig(f"{PLOTS}/02b_attendance_by_grade.png", dpi=130); plt.close()

# 2c. Which components move with the total? (correlation heatmap)
corr_cols = [ATT, "Midterm_Score", "Final_Score", "Assignments_Avg", "Quizzes_Avg",
             "Participation_Score", "Projects_Score", TOTAL]
plt.figure(figsize=(9, 7))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix"); plt.tight_layout()
plt.savefig(f"{PLOTS}/03_correlation_heatmap.png", dpi=130); plt.close()

# 2d. Department and prep-course comparison, score distribution.
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
df.groupby("Department")[TOTAL].mean().sort_values().plot.barh(ax=ax[0], color="teal")
ax[0].set_title("Mean Total Score by Department")
sns.boxplot(data=df, x="test_preparation_course", y=TOTAL, ax=ax[1])
ax[1].set_title("Effect of Prep Course")
sns.histplot(df[TOTAL], bins=30, kde=True, ax=ax[2]); ax[2].set_title("Total Score Distribution")
plt.tight_layout(); plt.savefig(f"{PLOTS}/04_groups_and_distribution.png", dpi=130); plt.close()

# =====================================================================
# 3. RISK DEFINITION  (our own criteria)
# =====================================================================
# Risk is defined from academic OUTCOMES (Total_Score and Grade), a transparent
# points system a teacher can audit:
#   Marks (Total_Score):  <60 -> 2 pts | 60-74 -> 1 pt | >=75 -> 0
#   Grade:                D or F -> 2  | C -> 1         | A or B -> 0
#   Total points: 0-1 = LOW, 2 = MEDIUM, 3-4 = HIGH
# Why outcomes and not attendance? Attendance is a BEHAVIOUR we want to use as
# an early-warning input. If it were also inside the label, the model would just
# re-learn our rule (data leakage). Keeping it on the input side lets the model
# answer the useful question: "who is likely to end up struggling, and why?"
# NOTE: in this file Total_Score and Grade are almost unrelated to each other
# (see EDA), so each is treated as an independent warning sign.
# Thresholds are assumptions - tune them to your institution's pass mark.
def risk_points(marks, grade):
    m = 2 if marks < 60 else 1 if marks < 75 else 0
    g = 2 if grade in ("D", "F") else 1 if grade == "C" else 0
    return m + g

def risk_label(pts):
    return "HIGH" if pts >= 3 else "MEDIUM" if pts == 2 else "LOW"

df["Risk"] = [risk_label(risk_points(m, g)) for m, g in zip(df[TOTAL], df["Grade"])]
print("\n=== Risk distribution (rule-based labels) ===")
print(df["Risk"].value_counts().to_string())
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Risk", order=["LOW", "MEDIUM", "HIGH"], palette=["green", "orange", "red"])
plt.title("Risk Distribution"); plt.tight_layout()
plt.savefig(f"{PLOTS}/05_risk_distribution.png", dpi=130); plt.close()

# =====================================================================
# 4. FEATURE SELECTION  (guarding against data leakage)
# =====================================================================
# The label is computed FROM Total_Score and Grade, so those must be excluded or
# the model just re-learns our rule. Final_Score is also dropped: we want to flag
# students BEFORE the final exam, when intervention can still help.
# Also excluded: IDs, names, email (identifiers) and gender (not relevant to
# academic risk, and using it would risk unfair bias).
# KEPT: Attendance, midterm, assignments, quizzes, participation, projects,
# subject scores, age, department, prep course - all known before the final.
numeric_features = ["Age", ATT, "Midterm_Score", "Assignments_Avg", "Quizzes_Avg",
                    "Participation_Score", "Projects_Score",
                    "math_score", "reading_score", "writing_score", "science_score"]
categorical_features = ["Department", "test_preparation_course"]
features = numeric_features + categorical_features

X, y = df[features], df["Risk"]
# Stratify so each split keeps the same Low/Medium/High proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Preprocessing lives INSIDE the pipeline so scaling/encoding is learned from
# training data only (no leakage into the test set).
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

# =====================================================================
# 5. MODELS + SELECTION
# =====================================================================
# class_weight="balanced": HIGH-risk students are usually the minority, and
# missing them is the costly error, so we up-weight rare classes.
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                            random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
print("\n=== 5-fold CV (macro F1 - treats all 3 risk classes equally) ===")
cv_scores = {}
for name, clf in candidates.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    cv_scores[name] = s.mean()
    print(f"{name:20s} {s.mean():.3f} (+/- {s.std():.3f})")
best_name = max(cv_scores, key=cv_scores.get)
print(f"-> Selected: {best_name}")

# =====================================================================
# 6. EVALUATION on the untouched test set
# =====================================================================
model = Pipeline([("prep", prep), ("clf", candidates[best_name])]).fit(X_train, y_train)
pred = model.predict(X_test)
labels = ["LOW", "MEDIUM", "HIGH"]
acc = accuracy_score(y_test, pred)
p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
print(f"\n=== Test-set results ({best_name}) ===")
print(f"Accuracy {acc:.3f} | Macro Precision {p:.3f} | Macro Recall {r:.3f} | Macro F1 {f:.3f}")
print(classification_report(y_test, pred, labels=labels, zero_division=0))
# For this use case RECALL on HIGH is the metric that matters most: a missed
# at-risk student gets no help, while a false alarm only costs a quick check-in.

# Sanity check: is the model better than guessing? A "most frequent class" dummy
# is the minimum bar any useful model must beat.
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
dp = dummy.predict(X_test)
print(f"Baseline (always predict '{y_train.mode()[0]}'): accuracy {accuracy_score(y_test, dp):.3f}, "
      f"macro F1 {precision_recall_fscore_support(y_test, dp, average='macro', zero_division=0)[2]:.3f}")
proba = model.predict_proba(X_test)
auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=model.classes_)
print(f"Model macro one-vs-rest ROC-AUC: {auc:.3f}  (0.5 = no better than chance)")

cm = confusion_matrix(y_test, pred, labels=labels)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}"); plt.tight_layout()
plt.savefig(f"{PLOTS}/06_confusion_matrix.png", dpi=130); plt.close()

# Feature importance (tree models) or coefficients magnitude (logistic).
clf = model.named_steps["clf"]
names = model.named_steps["prep"].get_feature_names_out()
if hasattr(clf, "feature_importances_"):
    imp = pd.Series(clf.feature_importances_, index=names)
else:
    imp = pd.Series(np.abs(clf.coef_).mean(axis=0), index=names)
imp.sort_values().tail(10).plot.barh(figsize=(7, 5), color="purple")
plt.title("Top Drivers of Risk Prediction"); plt.tight_layout()
plt.savefig(f"{PLOTS}/07_feature_importance.png", dpi=130); plt.close()

# =====================================================================
# 7. PREDICTION + RECOMMENDATION
# =====================================================================
def recommend(row, risk):
    """Rule-based advice that targets each student's actual weak spots."""
    tips = []
    if row[ATT] < 80:
        tips.append("Improve attendance" if row[ATT] < 65 else "Keep attendance above 80%")
    # (column, advice text, scale to convert to a 0-100 percentage)
    weak = {"Midterm_Score": ("midterm topics", 1), "Assignments_Avg": ("assignment completion", 1),
            "Quizzes_Avg": ("regular quiz practice", 1), "Projects_Score": ("project work", 1),
            "Participation_Score": ("class participation", 10)}   # participation is out of 10
    # flag components below 55%, worst first (max two so advice stays actionable)
    low = sorted([(row[k] * sc, v) for k, (v, sc) in weak.items() if row[k] * sc < 55])[:2]
    if low:
        tips.append("focus on " + " and ".join(v for _, v in low))
    if row["test_preparation_course"].lower() in ("none", "no", "not completed") and risk != "LOW":
        tips.append("consider joining the preparation course")
    if risk == "HIGH":
        tips.append("schedule an advisor meeting this week")
    if not tips:
        return "On track - keep up the good work."
    s = "; ".join(tips)
    return s[0].upper() + s[1:] + "."

def report_student(sid):
    """Print one student's card in the requested format."""
    row = df[df["Student_ID"] == sid].iloc[0]
    risk = row["Risk"]                                   # known from marks + grade
    early = model.predict(row[features].to_frame().T.astype(df[features].dtypes.to_dict()))[0]
    print(f"\nStudent: {sid}\nAttendance: {row[ATT]:.0f}%\nMarks: {row[TOTAL]:.0f} (Grade {row['Grade']})\n"
          f"Risk: {risk}   (model early-warning: {early})\nRecommendation:\n{recommend(row, risk)}")

# =====================================================================
# 8. AUTOMATED REPORT (bonus)
# =====================================================================
# Refit on ALL data for the final operational model (evaluation numbers above
# came from the held-out split; this just uses every student for deployment).
final_model = Pipeline([("prep", prep), ("clf", candidates[best_name])]).fit(X, y)
df["Model_Risk"] = final_model.predict(X)
# Headline risk = the rule-based risk, because for existing students the marks and
# grade are already known. Model_Risk is the early-warning view (what the model
# would have said using only pre-final information).
df["Predicted_Risk"] = df["Risk"]
df["Recommendation"] = [recommend(r, r["Predicted_Risk"]) for _, r in df.iterrows()]
df["Student"] = df["First_Name"].astype(str) + " " + df["Last_Name"].astype(str) + " (" + df["Student_ID"] + ")"
order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}                       # worst cases at the top
report = (df.assign(_o=df["Predicted_Risk"].map(order))
            .sort_values(["_o", ATT])
            [["Student", ATT, TOTAL, "Grade", "Predicted_Risk", "Model_Risk", "Recommendation"]]
            .rename(columns={ATT: "Attendance", TOTAL: "Marks", "Predicted_Risk": "Risk",
                             "Model_Risk": "Model_Early_Warning"}))
report["Attendance"] = report["Attendance"].round(0).astype(int).astype(str) + "%"
report.to_csv(f"{OUT}/risk_report.csv", index=False)
print(f"\nReport saved: {OUT}/risk_report.csv ({len(report)} students)")
print(report.head(5).to_string(index=False))

print("\n--- Sample student cards ---")
for sid in df.sort_values(TOTAL).iloc[[0, len(df)//2, -1]]["Student_ID"]:
    report_student(sid)
