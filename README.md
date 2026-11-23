# Hospital-Onset Infection Detection

Machine learning and language model experiments for predicting hospital-onset infection from patient admission data. Each patient record is converted into a short clinical text description, then modeled with classical ML (including stacking and blending ensembles) and fine-tuned transformers. SHAP is used to explain the best model.

## Notebooks

| File | What it does |
|---|---|
| `part1_ml_pipeline.ipynb` | Tabular to text conversion, TF-IDF features, chi-squared feature selection, classical models, hyperparameter tuning (GridSearchCV, RandomizedSearchCV, BayesSearchCV), stacking and blending, SMOTE and SMOTE-Tomek comparison, SHAP and ablation study on the best LightGBM model |
| `part2_nlp_llm.ipynb` | Fine-tuning of BioBERT, ClinicalBERT, BioBART and GPT-2 on the clinical text, plus a text augmentation re-run on ClinicalBERT |

## Data

The data is not included in this repository. The notebooks use the PhysioNet dataset "Predictors of Hospital Onset Infection: A Matched Retrospective Cohort Dataset", derived from the Mass General Brigham health system.

https://physionet.org/content/hospital-onset-infection

It is credentialed access, so you need to request access on PhysioNet and accept the data use agreement before downloading. Place the file as `HO_infxn_analysis.csv` in the same folder as the notebooks. If your copy has a different name, change `DATA_PATH` near the top of each notebook.

## Setup

```
pip install -r requirements.txt
```

The notebooks were run on Google Colab. Part 2 needs a GPU for reasonable training time. The random seed is fixed at 42.

## How to run

1. Place the dataset file next to the notebooks.
2. Run `part1_ml_pipeline.ipynb` from top to bottom.
3. Run `part2_nlp_llm.ipynb` from top to bottom.

## Dataset citation

Wei, Z., Sagers, L., McKenna, C., Pak, T., Rhee, C., Klompas, M., & Kanjilal, S. (2025). Predictors of Hospital Onset Infection: A Matched Retrospective Cohort Dataset. PhysioNet. https://doi.org/10.13026/k70x-0m81
