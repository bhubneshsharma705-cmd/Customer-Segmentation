# Customer Segmentation Project

## Overview
This project applies **K-Means clustering** to segment customers into distinct groups based on their demographics and purchasing behavior. By identifying these natural customer segments, businesses can design more targeted marketing strategies, personalize customer experiences, and improve retention and profitability.

## Objective
Rather than treating all customers as one group, this project analyzes customer data to uncover meaningful patterns — such as age, income, spending habits, and purchase frequency — that distinguish different types of customers from one another.

## Dataset Features
- **Age** – Customer's age
- **Income** – Annual income
- **Total_Spending** – Total amount spent
- **NumWebPurchases** – Number of purchases made online
- **NumStorePurchases** – Number of purchases made in-store
- **NumWebVisitsMonth** – Number of website visits per month
- **Recency** – Days since the customer's last purchase

## Methodology
1. **Data Preprocessing** – Cleaned the dataset and selected relevant features.
2. **Feature Scaling** – Applied `StandardScaler` to normalize feature ranges.
3. **Clustering** – Used the **K-Means** algorithm to group customers into segments.
4. **Model Deployment** – Built an interactive **Streamlit** web app for real-time segment prediction.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (K-Means, StandardScaler)
- Streamlit (web app)
- Jupyter Notebook (analysis)

## Project Files
| File | Description |
|---|---|
| `Analysis_Model.ipynb` | Exploratory data analysis and model training |
| `segmentaion.py` | Streamlit app for live segment prediction |
| `customer_segmentation.csv` | Dataset used for training |
| `kmeans_model.pkl` | Saved trained K-Means model |
| `scaler.pkl` | Saved StandardScaler used for preprocessing |

## How to Run
```bash
pip install -r requirements.txt
streamlit run segmentaion.py
```

## Outcome
This project demonstrates the complete data science workflow — from data cleaning and exploratory analysis to model building, evaluation, and deployment — showing how machine learning can be turned into a practical, usable business tool.
