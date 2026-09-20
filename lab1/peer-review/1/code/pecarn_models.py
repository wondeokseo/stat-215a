import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import pandas as pd
from sklearn.metrics import confusion_matrix

def train_decision_tree(X_train, y_train, max_depth, random_state=42):
    """

    Train a Decision Tree Classifier.
    Ensures that categorical variables are properly encoded for the decision tree.

    Parameters:
    - X_train: Features for training.
    - y_train: Target variable for training.
    - max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.

    Returns:
    - model: Trained Decision Tree Classifier.

    """
            
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight='balanced') # Positive ciTBI is rare, need to adjust for this.
    model.fit(X_train, y_train)
    return model

def train_xg_boost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
    """

    Train an XGBoost Classifier.
    Ensures that categorical variables are properly encoded for the XGBoost model.

    Parameters:
    - X_train: Features for training.
    - y_train: Target variable for training.
    - n_estimators: Number of boosting rounds.
    - learning_rate: Step size shrinkage used in update to prevent overfitting.
    - max_depth: Maximum depth of a tree.

    Returns:
    - model: Trained XGBoost Classifier.

    """

    ratio = np.sum(y_train == 0) / np.sum(y_train == 1)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        random_state=random_state, 
        enable_categorical=False,  # Treated as standard numeric features, allowing XGBoost to handle the NaNs natively
        scale_pos_weight=ratio,  # Positive ciTBI is rare, need to adjust for this.
    )
    model.fit(X_train, y_train)
    return model

def compute_sensitivity_npv(y_true, y_pred):
    """ 

    Compute the sensitivity and negative prediction value (NPV) for a binary classification model.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - sensitivity: The true positive rate.
    - npv: The negative predictive value.

    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = round(tp / (tp + fn), 5) if (tp + fn) > 0 else 0
    npv = round(tn / (tn + fn), 5) if (tn + fn) > 0 else 0
    return sensitivity, npv