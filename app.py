import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load the dataset
df = pd.read_csv(r'E:\Celebal project\Assignment4_featureengineering\tested.csv')  # change the path if necessary
print(df.head())

print(f"\nShape of dataset: {df.shape}")
# column types and not null counts
print("\nDataset Info:")
print(df.info())

# Summary stats
print("\nStatistical Summary:")
print(df.describe())

# Check null values
print("\nMissing Values:")
print(df.isnull().sum())

# Categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

#plotting
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=missing.values, y=missing.index, palette="magma")
plt.title("Missing Values Count")
plt.xlabel("Number of Missing Values")
plt.ylabel("Features")
plt.show()

df.drop('Cabin', axis=1, inplace=True)

# mode
print(df['Embarked'].mode())

# Fill null value with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Age'].fillna(df['Age'].median(), inplace=True)

print(df.isnull().sum())

num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[num_cols].describe()

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

#Outlier_detection
def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[col] < lower) | (data[col] > upper)]

for col in num_cols:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col} - Outliers: {len(outliers)}")

# Cap Fare at 99th percentile
fare_cap = df['Fare'].quantile(0.99)
df['Fare'] = df['Fare'].apply(lambda x: min(x, fare_cap))

corr_cols = ['Survived', 'Age', 'Fare', 'SibSp', 'Parch']
corr = df[corr_cols].corr()

#plotting
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[corr_cols], hue='Survived', diag_kind='kde')
plt.suptitle("Pairwise Plot with Survived", y=1.02)
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(x='Embarked', y='Survived', data=df)
plt.title("Survival Rate by Port of Embarkation")
plt.show()

# FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# IsAlone
# default = alone
df['IsAlone'] = 1  
df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

# Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Combining titles
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

# Binned Fare (optional if Fare is skewed)
df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[1, 2, 3, 4])

# Binned Age (optional for handling missing + non-linear age effect)
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 80], labels=[0,1,2,3,4])

label = LabelEncoder()

for col in ['Sex', 'Embarked', 'Title']:
    df[col] = label.fit_transform(df[col])

df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Define X and y
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_val)
y_prob = rf.predict_proba(X_val)[:, 1]

# Metrics
print("Accuracy:", accuracy_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_prob))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Optional: Confusion Matrix

sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Re-impute again before model loop
imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_imp, y_train)
# Predictions and evaluation
y_pred_lr = lr.predict(X_val_imp)
y_prob_lr = lr.predict_proba(X_val_imp)[:, 1]

from sklearn.metrics import accuracy_score, roc_auc_score

print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_val, y_prob_lr))

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(verbose=0)
}

for name, model in models.items():
    model.fit(X_train_imp, y_train)
    preds = model.predict(X_val_imp)
    probs = model.predict_proba(X_val_imp)[:, 1]
    
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_val, preds))
    print("ROC AUC:", roc_auc_score(y_val, probs))


# Sample a few instances for speed (optional)
X_sample = pd.DataFrame(X_val_imp, columns=X.columns).sample(
    n=min(100, len(X_val_imp)),
    random_state=42
)