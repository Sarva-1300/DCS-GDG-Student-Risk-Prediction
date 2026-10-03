# Student Academic Risk Prediction

An end-to-end Python pipeline that identifies students who may need academic
intervention, using attendance, assessment scores and grades.

## What it does
- **Cleans** messy data: duplicates, out-of-range values, inconsistent
  category spellings, missing values
- **Explores** relationships such as attendance vs. marks and attendance vs. grade,
  with 8 saved visualisations
- **Labels risk** (Low / Medium / High) with a transparent points system
  based on marks and grade
- **Trains and compares** Logistic Regression, Decision Tree and Random Forest
  using 5-fold cross-validation
- **Evaluates** with accuracy, precision, recall, F1, ROC-AUC, a confusion
  matrix and a majority-class baseline
- **Generates** a per-student report: Student | Attendance | Marks | Risk |
  Recommendation

## Key findings
- Attendance shows no linear relationship with Total_Score (r ≈ -0.01),
  but does relate to Grade (Spearman ρ ≈ 0.27)
- The dataset contains very little predictive signal, so the model performs
  only slightly above chance. See the evaluation section for details.

## Run it
pip install pandas scikit-learn seaborn matplotlib
python student_risk_pipeline.py dcs_student_data.csv

Outputs are written to `output/` (plots and `risk_report.csv`).

## Design notes
Attendance, Total_Score and Grade are kept out of the wrong side of the
model on purpose, to avoid data leakage. See the comments in the code.