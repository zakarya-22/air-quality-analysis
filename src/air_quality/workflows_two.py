"""
This file mirrors workflows.py but fixes the ordering and scoping issues.
Every function/class ends with 'Two' so it can coexist with the originals.
"""

from dataclasses import dataclass, field

from sklearn.linear_model import LinearRegression

from air_quality import data, evaluation, features, selection

DEFAULT_COLUMNS = [
    "city",
    "date",
    "hour",
    "site_latitude",
    "site_longitude",
    "pm2_5",
    "sulphurdioxide_so2_column_number_density",
    "carbonmonoxide_co_column_number_density",
    "nitrogendioxide_no2_column_number_density",
    "formaldehyde_tropospheric_hcho_column_number_density",
    "uvaerosolindex_absorbing_aerosol_index",
    "ozone_o3_column_number_density",
    "uvaerosollayerheight_aerosol_height",
    "cloud_cloud_fraction",
]

# Metadata columns that are never features � exclude them when scoping wide
EXCLUDED_RAW_COLUMNS = {"id", "site_id", "country", "month"}


@dataclass
class AdvancedPipelineConfigTwo:
    """Settings for run_advanced_two."""

    cities: list[str] = field(
        default_factory=lambda: ["Kampala", "Nairobi", "Lagos", "Bujumbura"]
    )
    columns: list[str] = field(default_factory=lambda: list(DEFAULT_COLUMNS))
    missing_threshold: float = 0.7
    use_wide_scope: bool = False  # True = all satellite columns (Module 4+)
    n_features_rfe: int = 8
    use_xgboost_tuning: bool = False  # True = Module 5 tuning on top


def run_advanced_two(config: AdvancedPipelineConfigTwo | None = None) -> dict:
    """
    Corrected Session 3 pipeline, building up module by module:

    Module 2: scope to all 4 cities, drop high-missing columns, fill per city,
              add temporal features, evaluate with GroupKFold (Module 3).
    Module 4: optionally use all satellite columns + RFE feature selection.
    Module 5: optionally add XGBoost hyperparameter tuning.

    Parameters:
    - config: pipeline settings; defaults are used if omitted

    Returns:
    - dict of metrics from evaluate_group_cv (and optionally tuning results)
    """
    config = config or AdvancedPipelineConfigTwo()
    train_df, _ = data.load_datasets()

    # --- Step 1: Scope ---
    if config.use_wide_scope:
        # Module 4+: use ALL satellite columns (exclude only metadata junk)
        columns = [c for c in train_df.columns if c not in EXCLUDED_RAW_COLUMNS]
    else:
        # Module 2-3: use the 8 hand-picked DEFAULT_COLUMNS
        columns = config.columns

    scoped = data.restrict_to_scope(train_df, config.cities, columns)

    # --- Step 2: Drop columns missing > threshold (Module 2) ---
    bad_columns = data.columns_above_missing_threshold(scoped, config.missing_threshold)
    cleaned = data.drop_columns(scoped, bad_columns)

    # --- Step 3: Fill remaining missing values per city (Module 2) ---
    # IMPORTANT: compute fillable_columns AFTER dropping, not before
    fillable_columns = [c for c in cleaned.columns if c not in ("city", "date")]
    cleaned = data.fill_missing_by_city(cleaned, fillable_columns)

    # --- Step 4: Add temporal features (Module 2) ---
    enriched = features.add_temporal_features(cleaned)

    # --- Step 5: Get feature columns ---
    feature_cols = features.feature_columns(enriched)

    # --- Step 6: Feature selection with RFE (Module 4) ---
    if config.use_wide_scope:
        feature_cols = selection.select_features_rfe(
            LinearRegression(), enriched, feature_cols, "pm2_5", config.n_features_rfe
        )

    # --- Step 7: Evaluate with GroupKFold (Module 3) ---
    model = LinearRegression()
    results = evaluation.evaluate_group_cv(
        model, enriched, feature_cols=feature_cols, groups_col="city", target_col="pm2_5"
    )

    # --- Step 8: Optionally add XGBoost tuning (Module 5) ---
    if config.use_xgboost_tuning:
        from air_quality import tuning  # lazy import � xgboost may not be installed

        param_grid = {"max_depth": [3, 5], "learning_rate": [0.05, 0.2]}
        tuning_results = tuning.tune_xgboost(
            enriched, feature_cols=feature_cols, param_grid=param_grid,
            target_col="pm2_5", groups_col="city"
        )
        results["tuning"] = tuning_results

    return results
