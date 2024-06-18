import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle
from sklearn.svm import SVC
from preprocess import preprocess_text
import os
from clean_data import clean_data

# Load and preprocess the dataset
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'skill_list.csv')

try:
    df = pd.read_csv(data_path)
    print("CSV file loaded successfully.")
    print(df.head())
except Exception as e:
    print(f"Error loading CSV file: {e}")

# Check the columns and data types
print("Columns in the dataframe:", df.columns)
print("Data types in the dataframe:", df.dtypes)

# Clean the data
df_cleaned = clean_data(df)
print("Class distribution after cleaning:")
print(df_cleaned['Label'].value_counts())

# Apply preprocessing
df_cleaned['Processed_Skill'] = df_cleaned['Skills'].apply(preprocess_text)

# Define X and y
X = df_cleaned['Processed_Skill']
y = df_cleaned['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a single pipeline with a placeholder for the classifier
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  
    ('classifier', LogisticRegression())
])

# Define a parameter grid that includes different classifiers and their hyperparameters
param_grid = [
    {
        'classifier': [LogisticRegression()],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [10, 50, 100],
        'classifier__max_features': ['auto', 'sqrt', 'log2']
    }, 
        {
        'classifier': [GradientBoostingClassifier()],
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5]
    }
]

# Adjust the cross-validation strategy based on class distribution
stratified_kfold = StratifiedKFold(n_splits=2)

# Perform Grid Search with the adjusted cross-validation strategy
grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, scoring='accuracy', refit=True)
grid_search.fit(X_train, y_train)

# Best parameters and estimator
print("Best parameters found: ", grid_search.best_params_)
print("Best estimator found: ", grid_search.best_estimator_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the best model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
