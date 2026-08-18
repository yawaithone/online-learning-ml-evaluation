# Evaluating the Effectiveness of Online Learning Technologies

**A Machine Learning-Driven Evaluation of Student Performance and Course Sentiment**

This project uses machine learning and NLP to (1) predict student performance in a virtual
learning environment and (2) classify learner sentiment from online course reviews. It was
completed for the *Data Mining & Machine Learning* module (MSc in Data Analytics, National
College of Ireland).

Authors: **Chanlin Naicker** & **Ya Wai Thone**

---

## Overview

Online and blended learning generate huge amounts of learner data, but understanding how that
data maps to *outcomes* and *satisfaction* is hard. This project tackles both sides:

- **Performance prediction (structured data):** a 4-class problem — predict whether a student
  will end a module as **Pass, Fail, Withdrawn, or Distinction** — using demographics,
  registration details, and aggregated VLE engagement.
- **Sentiment analysis (text data):** a 3-class problem — classify course reviews as
  **negative, neutral, or positive** — comparing a transformer model against a classical baseline.

The workflow follows the **KDD** process: sourcing → merging/cleaning → EDA → feature
engineering → modelling → evaluation.

---

## Datasets

> ℹ️ **The datasets are not included in this repository.** Download them from the official
> sources below and place them in the folders indicated, then run the notebooks.

### 1. OULAD (student performance)

The Open University Learning Analytics Dataset — 32,593 students across 22 module presentations,
split over 7 linked CSVs (demographics, registration, assessments, VLE interaction logs).

- **Download:** [analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset)
- Download the full dataset zip, extract it, and place the CSVs in **`oulad/data/`**:
  `studentInfo.csv`, `studentRegistration.csv`, `assessments.csv`, `studentAssessment.csv`,
  `courses.csv`, `vle.csv`, `studentVle.csv`
- License: CC-BY (Kuzilek et al., 2017)

### 2. Coursera Reviews (sentiment)

~99k English-language course reviews with 1–5 star ratings.

- **Download:** [github.com/sharmaroshan/Coursera-Reviews-Analysis](https://github.com/sharmaroshan/Coursera-Reviews-Analysis/blob/master/reviews.csv)
- Place `reviews.csv` (and `test_reviews.csv` if used) in **`text-analysis/data/`**

> If your local folder names differ, update the file paths in the first cell of each notebook.

---

## Methods

**OULAD – performance prediction**
- Merged 7 CSVs into a single student-level table; aggregated 10M+ VLE interaction rows into
  per-student engagement features (total clicks, active days, unique resources).
- One-hot encoding, `StandardScaler`, stratified 70/30 split, Chi-Square feature selection.
- Models: **Multinomial Logistic Regression**, **SVM (RBF)**, **Random Forest**.

**Coursera – sentiment analysis**
- Language detection (English only), text cleaning, 80/20 split.
- Models: **DistilBERT** (fine-tuned transformer) and **Multinomial Naïve Bayes** (TF-IDF baseline).
- Interpretability via **SHAP** on representative reviews.

---

## Results

### Student performance (4-class) — OULAD

| Model | Accuracy | Macro F1 | Cohen's κ | MCC | ROC AUC |
|---|---|---|---|---|---|
| Multinomial Logistic Regression | 0.602 | 0.587 | 0.452 | 0.458 | 0.863 |
| SVM (RBF) | 0.623 | 0.598 | 0.467 | 0.468 | 0.851 |
| **Random Forest** | **0.663** | **0.632** | **0.513** | **0.515** | **0.880** |

**Random Forest** performed best overall. VLE engagement features (total clicks, active days,
unique resources) and achieved marks were the strongest predictors.

### Course sentiment (3-class) — Coursera

| Model | Accuracy | Macro F1 | Cohen's κ | MCC |
|---|---|---|---|---|
| **DistilBERT** | **0.935** | 0.690 | 0.617 | 0.619 |
| Multinomial Naïve Bayes | 0.793 | 0.793 | 0.689 | 0.691 |

**DistilBERT** achieved the highest accuracy and strong contextual understanding; MNB was a
solid, balanced classical baseline.

---

## Repository structure

```
.
├── README.md
├── presentation/
│   └── DMML_slides.pptx
├── oulad/
│   ├── DMML_OULADProject.ipynb        # merging, EDA, LogReg / SVM / Random Forest
│   └── data/                          # <- place OULAD CSVs here (not tracked)
└── text-analysis/
    ├── DMML_DistilBERT_text.ipynb     # transformer sentiment classifier
    ├── DMML_Naive_Bayes_text.ipynb    # TF-IDF + Multinomial NB baseline
    ├── DMML_text_SHAP.ipynb           # SHAP interpretability
    └── data/                          # <- place Coursera reviews here (not tracked)
```

---

## How to run

```bash
# clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# (optional) create an environment
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn nltk langid \
            transformers torch shap wordcloud jupyter

# download the datasets (see "Datasets" above) into oulad/data/ and text-analysis/data/

# launch
jupyter notebook
```

Open any notebook and run top to bottom. The DistilBERT notebook is GPU-friendly (originally
trained in Google Colab).

---

## Tech stack

`Python` · `pandas` · `scikit-learn` · `Hugging Face Transformers (DistilBERT)` · `PyTorch` ·
`NLTK` · `SHAP` · `Matplotlib` / `Seaborn`

---

## License / attribution

OULAD is released under CC-BY (Kuzilek et al., 2017). The Coursera reviews dataset is sourced
from the linked public GitHub repository. This project is for academic and portfolio purposes.
