# Student Academic Risk Prediction

An end-to-end Python pipeline that identifies students who may need academic
intervention. It cleans the data, explores it, labels each student as
**Low / Medium / High** risk, compares three classification models, and
generates a per-student report with recommendations.

## Contents

| File | Purpose |
|---|---|
| `student_risk_pipeline.py` | The full pipeline (heavily commented) |
| `results.txt` | Full console output from the latest run (findings and metrics) |
| `risk_report.csv` | Automated report: Student, Attendance, Marks, Grade, Risk, Recommendation |
| `plots/` | 8 saved visualisations |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
python student_risk_pipeline.py dcs_student_data.csv
```

Outputs are written to `output/` (plots, `risk_report.csv`, `results.txt`).

## Dataset

10,030 rows and 21 columns: demographics, department, attendance, midterm and
final scores, assignment, quiz, participation and project scores, total score,
grade, prep-course completion and four subject scores.

## What the data showed

**Data quality issues found and fixed**
- 30 duplicate rows removed.
- About 200 missing values in each of 8 columns, filled with the median
  (numeric) or most common value (categorical).
- Impossible values set to missing: attendance above 100% or below 0 (3 rows),
  scores above 100 (4 rows), ages such as 87 (2 rows).
- The same category spelled several ways ("CS" / "Computer Science",
  "Math" / "Mathematics", " MALE " / "Male") was merged.
- Stray whitespace and tab characters in `math_score` were stripped.

**Attendance vs marks and grade**
- Attendance and `Total_Score` show **no relationship** (Pearson r = -0.011).
  Mean score is about 75 in every attendance band.
- Attendance and `Grade` **do** relate (Spearman rho = 0.27): Grade A students
  average about 80% attendance, while C, D and F students average about 71%.
- `Total_Score` is also unrelated to `Grade`, to the component scores
  (midterm, assignments and so on), to department and to the prep course.
  It appears to be randomly generated, which is a data-quality limitation
  worth keeping in mind.

## Risk criteria

Risk is built from academic **outcomes** using a transparent points system.

| Marks (`Total_Score`) | Points | Grade | Points |
|---|---|---|---|
| Below 60 | 2 | D or F | 2 |
| 60-74 | 1 | C | 1 |
| 75 or above | 0 | A or B | 0 |

Total points: **0-1 = LOW**, **2 = MEDIUM**, **3-4 = HIGH**.

Resulting distribution: 4,804 LOW, 3,187 MEDIUM, 2,009 HIGH.

Attendance is deliberately **not** part of the label. It is used as a model
input instead, so the model cannot simply re-learn the labelling rule
(data leakage). The thresholds are assumptions and should be tuned to a
real institution's pass mark.

## Model approach

- **Features:** age, attendance, midterm, assignments, quizzes, participation,
  projects, four subject scores, department, prep course.
- **Excluded:** `Total_Score` and `Grade` (they define the label),
  `Final_Score` (the goal is to flag students before the final exam),
  identifiers (ID, name, email) and gender (not relevant, avoids bias).
- **Models compared with 5-fold cross-validation (macro F1):**
  Logistic Regression 0.366, Decision Tree 0.341, Random Forest 0.287.
  Logistic Regression was selected. Class weighting was used so the smaller
  classes are not ignored.
- Preprocessing (imputation, scaling, encoding) sits inside a scikit-learn
  Pipeline, so nothing from the test set leaks into training.

## Model results (held-out test set, 2,000 students)

| Metric | Logistic Regression | Always predict LOW (baseline) |
|---|---|---|
| Accuracy | 0.40 | 0.48 |
| Macro F1 | 0.36 | 0.22 |
| Macro ROC-AUC | 0.58 | 0.50 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| LOW | 0.57 | 0.53 | 0.55 |
| MEDIUM | 0.33 | 0.14 | 0.20 |
| HIGH | 0.24 | 0.50 | 0.33 |

The confusion matrix is in `plots/06_confusion_matrix.png`.

## Limitations

The model performs only slightly above chance. This reflects the dataset, not
a coding fault: apart from the attendance-grade link, the columns carry almost
no predictive signal. The majority-class baseline has higher raw accuracy, but
the model is better at catching at-risk students (50% recall on HIGH), which
matters more for this use case. With real institutional data, where scores
move together, performance should improve substantially.

## Predictions and recommendations

Each student gets a card in this format:

```
Student: S6342
Attendance: 84%
Marks: 50 (Grade C)
Risk: HIGH   (model early-warning: LOW)
Recommendation: Focus on assignment completion; consider joining the
preparation course; schedule an advisor meeting this week.
```

The headline `Risk` comes from the rule-based criteria (marks and grade are
already known for existing students). `Model_Early_Warning` in the CSV shows
what the model predicts from pre-final information alone. Recommendations are
rule-based and target each student's weakest areas (attendance below 80%,
components below 55%, missing prep course).

## Possible improvements

- Tune risk thresholds with input from teaching staff.
- Try a binary "at risk or not" model, which may score better.
- Validate on real, non-synthetic student data.
