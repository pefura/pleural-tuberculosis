import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')
import joblib
import matplotlib.pyplot as plt


# Set colors for plots
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f', '#9b59b6']

# Set random seed for reproducibility
np.random.seed(42)

# Define the variables to use
categorical = ['sex', 'cough', 'fever', 'sweating', 'weight_loss', 'duration', 'hiv', 'cancer']
continuous = ['age', 'IMC', 'proteins', 'crp_pleu', 'ldh', 'PNN']

def prepare_data(df):
    # Convert categorical variables
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    for col in ['cough', 'fever', 'sweating', 'weight_loss', 'hiv', 'cancer']:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # Convert duration to ordinal
    duration_map = {'<= 1 month': 0, ']1, 2 month]': 1, ']2, 3 month]': 2, '> 3 month': 3}
    df['duration'] = df['duration'].map(duration_map)
    
    # Select features and target
    X = df[categorical + continuous]
    y = df['TBpleu']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle missing values
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cont_imputer = SimpleImputer(strategy='median')
    
    X_train_cat = cat_imputer.fit_transform(X_train[categorical])
    X_test_cat = cat_imputer.transform(X_test[categorical])
    
    X_train_cont = cont_imputer.fit_transform(X_train[continuous])
    X_test_cont = cont_imputer.transform(X_test[continuous])
    
    # Scale continuous variables
    scaler = StandardScaler()
    X_train_cont_scaled = scaler.fit_transform(X_train_cont)
    X_test_cont_scaled = scaler.transform(X_test_cont)
    
    # Combine categorical and continuous features
    X_train_processed = np.hstack((X_train_cat, X_train_cont_scaled))
    X_test_processed = np.hstack((X_test_cat, X_test_cont_scaled))
    
    return X_train_processed, X_test_processed, y_train, y_test

def plot_combined_learning_curves(models_dict, X, y):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models_dict.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=10, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[idx].plot(train_sizes, train_mean, label='Training score', color='red', marker='o')
        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='red')
        axes[idx].plot(train_sizes, val_mean, label='Cross-validation score', color='green', linestyle='--', marker='s')
        axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='green')
        
        axes[idx].set_xlabel('Training Examples')
        axes[idx].set_ylabel('ROC AUC Score')
        axes[idx].set_title(f'Learning Curves - {name}')
        axes[idx].legend(loc='lower right')
        axes[idx].grid(True)
    
    # Remove the last subplot if it exists
    if len(axes) > len(models_dict):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('combined_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_roc_curves(results_dict, y_test):
    plt.figure(figsize=(10, 8))
    
    for idx, (name, model_results) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, model_results['probas'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('combined_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'predictions': y_pred,
        'probas': y_pred_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return results

def save_classification_reports(results_dict, y_test):
    with open('classification_reports.txt', 'w') as f:
        f.write("Classification Reports for Test Set\n")
        f.write("="*50 + "\n\n")
        
        for name, model_results in results_dict.items():
            f.write(f"{name}\n")
            f.write("-"*50 + "\n")
            report = classification_report(y_test, model_results['predictions'])
            f.write(report)
            f.write("\nAUC Score: {:.3f}\n\n".format(model_results['auc']))
            f.write("\n")

def calculate_cv_auc_scores(models_dict, X, y, cv=5):
    """
    Calculate cross-validation AUC scores with standard deviation for multiple models.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of models to evaluate
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Dictionary containing mean AUC scores and standard deviations for each model
    """
    results = {}
    
    for name, model in models_dict.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        mean_auc = cv_scores.mean()
        std_auc = cv_scores.std()
        results[name] = {
            'mean_auc': mean_auc,
            'std_auc': std_auc
        }
        print(f"{name}:")
        print(f"Mean AUC: {mean_auc:.3f} (±{std_auc:.3f})")
    
    return results

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('pleural_effusion_ML.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Define models with increased iterations/convergence parameters
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': randint(100, 500),
                'max_depth': [None] + list(range(5, 31)),
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5),
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(eval_metric='logloss', random_state=42),
            'params': {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=2000, random_state=42),
            'params': {
                'C': uniform(0.1, 10),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': uniform(0.1, 10),
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5)),
                'class_weight': ['balanced', None]
            }
        },
        'MLP': {
            'model': MLPClassifier(max_iter=2000, random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': uniform(0.0001, 0.01),
                'learning_rate_init': uniform(0.001, 0.01),
                'early_stopping': [True]
            }
        }
    }
    
    # Initialize models
    best_models = {}
    for name, model_info in models.items():
        best_models[name] = model_info['model']
    
    # Train and save models
    for name, model in best_models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        file_name = name.lower().replace(' ', '_')
        joblib.dump(model, f'models/{file_name}.joblib')
        
    # Save preprocessors
    preprocessors = {
        'cat_imputer': SimpleImputer(strategy='most_frequent'),
        'cont_imputer': SimpleImputer(strategy='median'),
        'scaler': StandardScaler()
    }
    
    # Fit and save preprocessors
    cat_data = df[categorical]
    cont_data = df[continuous]
    
    preprocessors['cat_imputer'].fit(cat_data)
    preprocessors['cont_imputer'].fit(cont_data)
    preprocessors['scaler'].fit(cont_data)
    
    for name, preprocessor in preprocessors.items():
        joblib.dump(preprocessor, f'models/{name}.joblib')
    
    # Perform RandomizedSearchCV
    results = {}
    print("\nModel Performance Summary:")
    print("="*80)
    
    for name, model_info in models.items():
        print(f"\n{name}:")
        print("-"*50)
        
        random_search = RandomizedSearchCV(
            model_info['model'],
            model_info['params'],
            n_iter=50,
            cv=5,
            n_jobs=-1,
            scoring='roc_auc',
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # Evaluate model
        results[name] = evaluate_model(best_model, X_test, y_test, name)
        
        # Print detailed results
        print("\nBest Parameters:")
        print(random_search.best_params_)
        print(f"\nBest Cross-validation Score (AUC): {random_search.best_score_:.3f}")
        print(f"Test Set AUC: {results[name]['auc']:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, results[name]['predictions']))
    
    # Create combined visualizations
    plot_combined_learning_curves(best_models, X_train, y_train)
    plot_combined_roc_curves(results, y_test)
    
    # Save classification reports
    save_classification_reports(results, y_test)
    
    # Calculate cross-validation AUC scores
    cv_auc_results = calculate_cv_auc_scores(best_models, X_train, y_train)
    
    # Print final comparison
    print("\nFinal Model Comparison:")
    print("="*50)
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': results[name]['accuracy'],
            'Precision': results[name]['precision'],
            'Recall': results[name]['recall'],
            'F1-score': results[name]['f1'],
            'AUC': results[name]['auc']
        } for name in models.keys()
    }).round(3)
    
    print(comparison_df.T.to_string())
    
    # Save results
    comparison_df.T.to_csv('model_comparison_results.csv')


rf_best_models = joblib.load('models/random_forest.joblib')

rf_feature_importances = rf_best_models.feature_importances_

rf_feature_importances


# Define categorical and continuous features
categorical_rf = ['sex', 'cough', 'fever', 'sweating', 'weight loss', 'duration', 'hiv', 'cancer']
continuous_rf = ['age', 'BMI (kg/m^2)', 'pleural proteins', 'pleural CRP', 'pleural LDH', 'pleural PMN']
features = categorical_rf + continuous_rf

# Combine features and importances into a DataFrame
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances as a bar chart
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title("Feature Importance - Random Forest", fontsize=14)
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.tight_layout()

# Show the plot
plt.show()


df = pd.read_csv('pleural_effusion_ML.csv')
print(df.head())  # Print the DataFrame to see the data   

df["PNN"].describe()

import sys
print(sys.version)

import sklearn
print(sklearn.__version__)