# Customer Churn Root-Cause and Risk Scoring Analysis

![Customer Churn Analytics Dashboard](Customer%20Churn%20Analytics%20Dashboard.png)

**Live App:** [Try the Churn Prediction App](https://telecom-churn-risk-analysis-tbdvpp8c83pedzxnnwk3ab.streamlit.app/)

## 1. Project Overview

This project analyzes customer churn patterns, identifies characteristics associated with churn, compares machine learning models, and generates customer churn-risk scores to support data-driven retention decisions.

The goal is to help telecom businesses identify customers who may be at risk of leaving and prioritize retention efforts.

## 2. Business Problem

Customer churn can reduce revenue and increase customer acquisition costs. Understanding which customers are more likely to leave can help businesses develop targeted retention strategies.

This project addresses three questions:

- What customer characteristics are associated with churn?
- Which machine learning model best identifies customers at risk of leaving?
- How can churn predictions support customer retention decisions?

## 3. Dataset

The project uses the IBM Telco Customer Churn dataset.

- **Records:** 7,043 customers
- **Target variable:** `Churn`
- **Features:** Customer demographics, subscribed services, contract details, billing information, and account tenure.

The target variable indicates whether a customer left the service (`Yes`) or stayed (`No`).

## 4. Methodology

The project follows the CRISP-DM framework.

1. **Business Understanding:** Define the churn problem and business objectives.
2. **Data Understanding:** Explore customer characteristics, churn distribution, and relationships between variables.
3. **Data Preparation:** Handle missing values, inspect duplicates, and prepare numerical and categorical features.
4. **Modeling:** Train and compare Logistic Regression, Decision Tree, Random Forest, and XGBoost models.
5. **Evaluation:** Compare model performance using precision, recall, F1-score, accuracy, and ROC-AUC.
6. **Deployment:** Save the selected model pipeline and build a Streamlit application for churn-risk prediction.

## 5. Exploratory Data Analysis

The dataset contains **7,043 customers**, of whom **1,869 churned**, representing approximately **26.5%** of the customer base.

The analysis explores:

- Churn distribution.
- Customer tenure and monthly charges.
- Contract types and churn rates.
- Customer services associated with churn.
- Relationships between customer characteristics and churn.

## 6. Model Performance

The following results were recorded during model evaluation:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% | 84.13% |
| Decision Tree | 75.51% | 52.69% | 75.94% | 62.21% | 83.23% |
| Random Forest | 76.37% | 54.85% | 62.03% | 58.22% | 82.25% |
| Regularized Random Forest | 75.51% | 52.53% | 80.48% | 63.57% | 84.33% |
| XGBoost | 80.13% | 65.67% | 52.67% | 58.46% | 84.47% |
| Tuned Random Forest | 74.95% | 51.85% | 78.88% | 62.57% | 84.22% |

The **Regularized Random Forest** was selected for the deployed application because it achieved the highest recall and F1-score among these evaluated models. Its recall indicates that it identified approximately 80.5% of actual churners in the test set.

These metrics are based on the notebook's recorded test results. The final model and evaluation procedure should remain consistent when the notebook is updated.

## 7. Key Findings

- Customers on month-to-month contracts had a churn rate of approximately **42.7%**.
- Customers on one-year contracts had a churn rate of approximately **11.3%**.
- Customers on two-year contracts had a churn rate of approximately **2.8%**.
- Contract type, tenure, total charges, online security, and technical support were among the important model features.

These findings show associations with churn; they do not establish that any single factor directly causes a customer to leave.

## 8. Business Recommendations

Based on the observed patterns, telecom businesses could:

- Prioritize at-risk customers for proactive retention outreach.
- Investigate the needs of customers on month-to-month contracts.
- Review onboarding and engagement strategies for newer customers.
- Consider whether security and technical support offerings address customer needs.
- Use churn probabilities to help prioritize retention resources.

Retention campaigns should be tested to determine whether they actually reduce churn.

## 9. Streamlit Application

The deployed application allows users to enter customer details and receive a predicted churn probability and risk classification.

**Try the app:** [Telecom Churn Risk Prediction](https://telecom-churn-risk-analysis-tbdvpp8c83pedzxnnwk3ab.streamlit.app/)

## 10. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## 11. Project Structure

```text
telecom-churn-risk-analysis/
├── churn_root_cause_pipeline.ipynb
├── app.py
├── requirements.txt
├── Customer Churn Analytics Dashboard.png
└── models/
    └── churn_random_forest_pipeline.joblib
```

## 12. Run Locally

**Prerequisites:** Python and Git.

1. Clone the repository:

   ```bash
   git clone https://github.com/arnokboi-commits/telecom-churn-risk-analysis.git
   ```

2. Enter the project directory:

   ```bash
   cd telecom-churn-risk-analysis
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in your terminal.

The saved model file must be present at `models/churn_random_forest_pipeline.joblib`.

## 13. Limitations

- The dataset represents a particular telecom customer population and may not reflect every business.
- Model predictions are estimates, not guarantees of customer behavior.
- Feature importance describes predictive relationships, not causal effects.
- Model performance may change when applied to new customer data.

## 14. Future Improvements

- Validate the model on new customer data.
- Select the classification threshold using validation data.
- Monitor model performance after deployment.
- Test retention strategies through controlled experiments.
- Improve the application with batch scoring and downloadable results.

## 15. Repository

**GitHub:** [telecom-churn-risk-analysis](https://github.com/arnokboi-commits/telecom-churn-risk-analysis)



