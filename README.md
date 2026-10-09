# credit_card_fraud_detection

Credit Card Fraud Detection
An end-to-end Machine Learning project implemented in a Jupyter Notebook to detect fraudulent credit card transactions. The project uses Pandas and NumPy for data processing, Matplotlib and Seaborn for visual analysis, StandardScaler for feature scaling, and compares Logistic Regression and Random Forest Classifier models evaluated via ROC-AUC and confusion matrices. The trained model is serialized using Joblib.

Table of Contents
Overview

Tech Stack

Notebook Workflow

Repository Structure

Installation & Setup

Running the Notebook

Model Performance & Metrics

License

Overview
Credit card fraud represents a major financial challenge due to extreme class imbalance (fraudulent transactions make up a tiny fraction of total activity). This project addresses data scaling and imbalanced classification using baseline Logistic Regression and ensemble Random Forest Classifier models, evaluating performance with ROC-AUC curves and detailed classification metrics.

Tech Stack
Environment: Jupyter Notebook / JupyterLab

Data Manipulation: pandas, numpy

Data Visualization: matplotlib, seaborn

Machine Learning (Scikit-Learn):

Preprocessing: StandardScaler

Data Splitting: train_test_split

Classification Algorithms: LogisticRegression, RandomForestClassifier

Model Evaluation: accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

Model Persistence: joblib

Notebook Workflow
All analysis and modeling steps are executed sequentially within a single notebook (credit_card_fraud_detection.ipynb):

Exploratory Data Analysis (EDA): Inspects transaction distributions, class imbalance between legitimate and fraudulent transactions, and feature correlations using seaborn and matplotlib.

Data Preprocessing & Scaling: Applies StandardScaler to normalize skewed numeric features (such as Amount and Time) and splits data into training and test sets via train_test_split.

Model Training:

Fits a baseline Logistic Regression model.

Fits an ensemble Random Forest Classifier model to capture non-linear decision boundaries.

Model Evaluation & Visualization:

Computes Accuracy, Precision, Recall, and F1-Score via classification_report.

Plots Confusion Matrices to analyze False Positives and False Negatives.

Generates and plots ROC Curves and computes ROC-AUC Scores (roc_auc_score, roc_curve) to assess model discrimination under class imbalance.

Model Serialization: Uses joblib to save the trained model and scaler to disk for deployment or future inference.

Repository Structure
Plaintext
credit-card-fraud-detection/
├── credit_card_fraud_detection.ipynb   # Single Jupyter Notebook with full analysis & pipeline
├── credit_card_model.joblib            # Serialized model artifact saved via Joblib
├── scaler.joblib                      # Serialized StandardScaler artifact
├── requirements.txt                    # List of Python dependencies
└── README.md                           # Project documentation
Installation & Setup
Clone the repository:

Bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

Bash
pip install -r requirements.txt
Running the Notebook
Launch Jupyter directly from your terminal:

Bash
jupyter notebook credit_card_fraud_detection.ipynb
Or via JupyterLab:

Bash
jupyter lab
Execute all cells sequentially (Cell -> Run All) to run data preprocessing, train both classifiers, display the ROC-AUC curves, and save the .joblib model file.

Model Performance & Metrics
Because credit card datasets are heavily imbalanced, overall Accuracy is insufficient. Models are primarily evaluated on Recall (catching as many fraudulent cases as possible) and ROC-AUC Score:

Logistic Regression: Serves as a fast, highly interpretable baseline model.

Random Forest Classifier: Captures complex feature interactions, typically achieving higher ROC-AUC scores and lower False Negative rates.

ROC Curves: Plotted directly in the notebook using matplotlib to visually compare decision thresholds across models.

License
Distributed under the MIT License. See LICENSE for details.
