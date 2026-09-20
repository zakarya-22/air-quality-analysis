"""Evaluation utilities for the Air Quality pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """
    Compute RMSE, MAE and R2 for a set of predictions.

    Parameters:
    - y_true: true target values
    - y_pred: predicted target values, same length as y_true

    Returns:
    - dict with keys "rmse", "mae", "r2"
    """
    # TODO: compute RMSE (the square root of mean_squared_error), MAE and R2,
    # and return them in a dict with keys "rmse", "mae", "r2"

    rmse = mean_squared_error(y_true,y_pred)**0.5
    mae = mean_absolute_error(y_true,y_pred)
    r2= r2_score(y_true,y_pred)
    L=[rmse,mae,r2]
    d={"rmse":rmse,"mae":mae,"r2":r2}
    return d


def evaluate_manual_split(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "pm2_5",
) -> dict[str, float]:
    """
    Fit a model on train_df and evaluate it on test_df (a different city).

    Parameters:
    - model: a scikit-learn estimator, following the fit/predict contract
    - train_df: DataFrame used to fit the model
    - test_df: DataFrame used to evaluate the model (a city train_df has not seen)
    - feature_cols: list of column names used as input features
    - target_col: name of the column to predict

    Returns:
    - dict of regression metrics (see regression_metrics), computed on test_df
    """
    # TODO: fit model on train_df's features/target, predict on test_df's
    # features, then return regression_metrics computed against test_df's
    # true target
    
    model.fit(train_df[feature_cols],train_df[target_col])
    predictions= model.predict(test_df[feature_cols])

    return regression_metrics(test_df[target_col],predictions)


# ============================================================================
# Session 3 — optional modules (not needed for Session 2's practical_3)
# ============================================================================

# --- Module 3: Robust evaluation (content/session3/practical_3) ---


def evaluate_group_cv(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    groups_col: str = "city",
    target_col: str = "pm2_5",
) -> dict:
    """
    Evaluate a model with GroupKFold, holding out one group at a time.

    Generalizes evaluate_manual_split's single train/test pair to every group in
    df: with as many folds as there are distinct values of groups_col, each fold
    leaves one entire group out for validation and trains on the rest — the same
    guarantee (a group's rows never appear in both train and validation) applied
    systematically instead of to a single manually chosen pair.

    Parameters:
    - model: a scikit-learn estimator, following the fit/predict contract; a
      fresh clone() is fitted for each fold so folds cannot leak state into
      one another
    - df: DataFrame containing feature_cols, target_col and groups_col
    - feature_cols: list of column names used as input features
    - groups_col: column identifying each row's group (e.g. "city")
    - target_col: name of the column to predict

    Returns:
    - dict with "folds" (one dict per fold, each with "test_group" and that
      fold's regression metrics) and "rmse_mean"/"rmse_std"/"mae_mean"/
      "mae_std"/"r2_mean"/"r2_std" aggregated across folds
    """
    # TODO: run GroupKFold with one fold per distinct value of groups_col (so
    # each fold holds out exactly one group), fit a fresh clone() of model on
    # each fold's training rows, evaluate it on the held-out group with
    # regression_metrics, then aggregate the mean and std of each metric
    # across folds
    
    x= df[feature_cols]
    y= df[target_col]
    d={}
    d["folds"]= []    
    n= df[groups_col].nunique()
    group_kfold = GroupKFold(n_splits=n)
    groups= df[groups_col]
    
    for (train_idx, test_idx) in group_kfold.split(x, y, groups=groups):
        
                        
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        cloned_model= clone(model)
        cloned_model.fit(x_train, y_train)
        predictions= cloned_model.predict(x_test)
        
        fold_dict= regression_metrics( y_test , predictions )  
        fold_dict["test_group"]= df[groups_col].iloc[test_idx[0]]
        
        d["folds"].append(fold_dict)
    
    folds_df = pd.DataFrame(d["folds"])
    d["r2_mean"]= float(folds_df['r2'].mean())
    d["rmse_mean"]= float(folds_df['rmse'].mean())
    d["mae_mean"]= float(folds_df['mae'].mean())
    d["mae_std"]= float(folds_df['mae'].std())
    d["rmse_std"]= float(folds_df['rmse'].std())
    d["r2_std"]= float(folds_df['r2'].std())
    
    return d

