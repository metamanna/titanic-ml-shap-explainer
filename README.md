# titanic-ml-shap-explainer
Predicting Titanic survival using ML models (Random Forest, XGBoost, CatBoost, Logistic Regression) with advanced feature engineering and SHAP explainability to visualize feature importance and model decisions.
# 🚢 Titanic Survival Prediction with SHAP Explainability

A complete machine learning pipeline for predicting survival on the Titanic using feature engineering, model evaluation, and explainability via SHAP values.

---

# 📁 Project Structure

Assignment4_featureengineering/

1. app.py # Main script for training, evaluating, and explaining models
2. tested.csv # Titanic dataset (input)
3. README.md # Project documentation

---

# 📌 Project Highlights

- ✅ Cleaned Titanic dataset
- 🔍 Feature engineering: FamilySize, IsAlone, Title, FareBin, AgeBin
- 📊 Data visualization with seaborn and matplotlib
- 🤖 Model training with:
  - Random Forest  
  - Logistic Regression  
  - Gradient Boosting  
  - XGBoost  
  - CatBoost
- 📈 Evaluation metrics: Accuracy, ROC AUC, Confusion Matrix, Classification Report
- 💡 Explainability using SHAP (summary plot, bar plot, force plot)

---

# Setup Instructions

 1. Clone the Repository

```bash
git clone https://github.com/yourusername/titanic-shap-survival.git
cd titanic-shap-survival
2. Create a Virtual Environment (Optional but Recommended)
bash
Copy
Edit
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the Script
Make sure tested.csv is in the same folder as app.py.

bash
Copy
Edit
python app.py
📦 Requirements
The requirements.txt file should contain:

txt
Copy
Edit
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
catboost
shap
You can generate it with:

bash
Copy
Edit
pip freeze > requirements.txt
📊 SHAP Explainability Examples
Feature importance (bar and summary plots)

SHAP force plots for individual predictions

Helps interpret why a model predicted survival or not

📈 Example Outputs
Accuracy and ROC AUC for each model

Confusion matrix heatmaps

SHAP summary plots

📌 Future Work
Streamlit dashboard integration

Upload your own dataset for prediction

Export model insights as PDF reports

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

📄 License
MIT License

🧠 Author
Tamanna Aggarwal
