"""
This module defines paths, feature sets and helper functions for analyzing and modeling
data related to endometriosis prediction.
"""

import pandas as pd
import numpy as np

# Unprepared data path
ENDO_DATA_GFORMS_PATH = "./real_data/womens_health_research_answer.csv"

# Prepared data path
ENDO_DATA_PREDICTION_PATH = "./real_data/preds_preprocessed_endo_data.csv"

# All features that are availale for early prediction of endometriosis
FEATURES_BASELINE_PREDICTION = [
    "age_18_24",
    "age_25_34",
    "age_35_44",
    "age_45_54",
    "was_pregnant",
    "has_thyroid_disorder",
    "experienced_infertility",
    "has_anemia",
    "BMI",
    "pelvic_pain_before_period",
    "pelvic_pain_after_period",
    "pelvic_pain_days_during_period",
    "pelvic_pain_frequency_between_periods",
    "pelvic_pain_worst",
    "pelvic_pain_average",
    "lower_back_pain_before_period",
    "lower_back_pain_after_period",
    "lower_back_pain_days_during_period",
    "lower_back_pain_frequency_between_periods",
    "lower_back_pain_worst",
    "lower_back_pain_average",
    "headache_before_period",
    "headache_after_period",
    "headache_days_during_period",
    "headache_frequency_between_periods",
    "headache_worst",
    "headache_average",
    "bloating_during_period",
    "bloating_between_periods",
    "pain_in_legs_hips_during_period",
    "pain_in_legs_hips_between_periods",
    "fatigue_during_period",
    "fatigue_between_periods",
    "pain_after_sex",
    "pelvic_pain_during_intercourse",
    "deep_vaginal_pain_during_intercourse",
    "not_sex_active",
    "regular_bowel_movements",
    "painful_bowel_movements",
    "difficulty_controlling_urination",
    "pain_during_urination",
    "3_4days_period_duration",
    "5_6days_period_duration",
    "7plus_days_period_duration",
    "24_31d_cycle_length",
    "32_38d_cycle_length",
    "39_50d_cycle_length",
    "more51d_cycle_length",
    "too_irregular_cycle_length",
    "bleeding_duration_changes",
    "large_blood_clots_frequency",
    "spotting_between_periods",
    "heavy_bleeding_frequency",
    "night_time_changes",
    "family_history_endometriosis_prediction",
    "family_history_infertility",
    "family_history_heavy_bleeding",
    "family_history_pelvic_pain",
    "bleeding_impact_social_events",
    "bleeding_impact_home_jobs",
    "bleeding_impact_work",
    "bleeding_impact_physical_activity",
    "pain_impact_social_events",
    "pain_impact_home_jobs",
    "pain_impact_work",
    "pain_impact_physical_activity",
    "pain_impact_appetite",
    "pain_impact_sleep",
    "experiences_mood_swings",
    "unable_to_cope_with_pain",
    "takes_over_cntr_pills",
]

# All features that can be used for PU learning when predicting the endometriosis in the family
PU_PREDICTION_FEATURES = [
    "age_18_24",
    "age_25_34",
    "age_35_44",
    "age_45_54",
    "was_pregnant",
    "has_thyroid_disorder",
    "experienced_infertility",
    "has_anemia",
    "BMI",
    "pelvic_pain_before_period",
    "pelvic_pain_after_period",
    "pelvic_pain_days_during_period",
    "pelvic_pain_frequency_between_periods",
    "pelvic_pain_worst",
    "pelvic_pain_average",
    "lower_back_pain_before_period",
    "lower_back_pain_after_period",
    "lower_back_pain_days_during_period",
    "lower_back_pain_frequency_between_periods",
    "lower_back_pain_worst",
    "lower_back_pain_average",
    "headache_before_period",
    "headache_after_period",
    "headache_days_during_period",
    "headache_frequency_between_periods",
    "headache_worst",
    "headache_average",
    "bloating_during_period",
    "bloating_between_periods",
    "pain_in_legs_hips_during_period",
    "pain_in_legs_hips_between_periods",
    "fatigue_during_period",
    "fatigue_between_periods",
    "pain_after_sex",
    "pelvic_pain_during_intercourse",
    "deep_vaginal_pain_during_intercourse",
    "not_sex_active",
    "regular_bowel_movements",
    "painful_bowel_movements",
    "difficulty_controlling_urination",
    "pain_during_urination",
    "3_4days_period_duration",
    "5_6days_period_duration",
    "7plus_days_period_duration",
    "24_31d_cycle_length",
    "32_38d_cycle_length",
    "39_50d_cycle_length",
    "more51d_cycle_length",
    "too_irregular_cycle_length",
    "bleeding_duration_changes",
    "large_blood_clots_frequency",
    "spotting_between_periods",
    "heavy_bleeding_frequency",
    "night_time_changes",
    "family_history_endometriosis",
    "family_history_infertility",
    "family_history_heavy_bleeding",
    "family_history_pelvic_pain",
    "bleeding_impact_social_events",
    "bleeding_impact_home_jobs",
    "bleeding_impact_work",
    "bleeding_impact_physical_activity",
    "pain_impact_social_events",
    "pain_impact_home_jobs",
    "pain_impact_work",
    "pain_impact_physical_activity",
    "pain_impact_appetite",
    "pain_impact_sleep",
    "experiences_mood_swings",
    "unable_to_cope_with_pain",
    "takes_over_cntr_pills",
]

# Includes features that were excluded to prevent multicolinearity and data leakage
ALL_AVAILABLE_FEATURES = [
    "age_under_18",
    "age_18_24",
    "age_25_34",
    "age_35_44",
    "age_45_54",
    "was_pregnant",
    "has_thyroid_disorder",
    "experienced_infertility",
    "has_anemia",
    "BMI",
    "pelvic_pain_before_period",
    "pelvic_pain_after_period",
    "pelvic_pain_days_during_period",
    "pelvic_pain_frequency_between_periods",
    "pelvic_pain_worst",
    "pelvic_pain_average",
    "lower_back_pain_before_period",
    "lower_back_pain_after_period",
    "lower_back_pain_days_during_period",
    "lower_back_pain_frequency_between_periods",
    "lower_back_pain_worst",
    "lower_back_pain_average",
    "headache_during_period",
    "headache_before_period",
    "headache_after_period",
    "headache_days_during_period",
    "headache_frequency_between_periods",
    "headache_worst",
    "headache_average",
    "bloating_during_period",
    "bloating_between_periods",
    "pain_in_legs_hips_during_period",
    "pain_in_legs_hips_between_periods",
    "fatigue_during_period",
    "fatigue_between_periods",
    "pain_after_sex",
    "pelvic_pain_during_intercourse",
    "deep_vaginal_pain_during_intercourse",
    "not_sex_active",
    "regular_bowel_movements",
    "painful_bowel_movements",
    "difficulty_controlling_urination",
    "pain_during_urination",
    "1_2days_period_duration",
    "3_4days_period_duration",
    "5_6days_period_duration",
    "less24d_cycle_length",
    "24_31d_cycle_length",
    "32_38d_cycle_length",
    "39_50d_cycle_length",
    "more51d_cycle_length",
    "too_irregular_cycle_length",
    "bleeding_duration_changes",
    "large_blood_clots_frequency",
    "spotting_between_periods",
    "heavy_bleeding_frequency",
    "night_time_changes",
    "bleeding_heaviness",
    "family_history_endometriosis",
    "family_history_endometriosis_prediction",
    "family_history_infertility",
    "family_history_heavy_bleeding",
    "family_history_pelvic_pain",
    "bleeding_impact_social_events",
    "bleeding_impact_home_jobs",
    "bleeding_impact_work",
    "bleeding_impact_physical_activity",
    "pain_impact_social_events",
    "pain_impact_home_jobs",
    "pain_impact_work",
    "pain_impact_physical_activity",
    "pain_impact_appetite",
    "pain_impact_sleep",
    "experiences_mood_swings",
    "unable_to_cope_with_pain",
    "takes_over_cntr_pills",
    "suggestions_questions",
    "has_fibroids",
    "had_hysterectomy",
    "takes_presc_painkillers",
    "takes_hormones_for_pain",
    "all_conditions_mentioned",
    "is_pregnant",
    "takes_hormones_only_for_contracep",
    "has_endometriosis",
]


def get_normalized_shap_table(
    shap_importance_rf,
    shap_importance_xgb,
    shap_importance_ada,
    shap_importance_mlp,
    shap_importance_tabpfn,
):
    """
    Normalizes SHAP importance values for multiple models and merges them into a single table.

    This function applies min-max scaling to SHAP importance scores from different models
    (Random Forest, XGBoost, AdaBoost, MLP and TabPFN). The normalized scores are then merged
    into a single table, and the mean importance score across all models is calculated.

    Args:
        shap_importance_rf: SHAP importance scores from the Random Forest model.
        shap_importance_xgb: SHAP importance scores from the XGBoost model.
        shap_importance_ada: SHAP importance scores from the AdaBoost model.
        shap_importance_mlp: SHAP importance scores from the MLP model.
        shap_importance_tabpfn: SHAP importance scores from the TabPFN model.

    Returns:
        A merged table containing normalized SHAP importance scores for each feature,
                      as well as the mean importance score across all models.
    """

    def normalize_importance(df, model_name):
        """Applies min-max normalization to the importance scores of a given model."""
        col_name = f"norm_importance_{model_name}_shap"
        df[col_name] = (df["importance"] - df["importance"].min()) / (
            df["importance"].max() - df["importance"].min()
        )
        return df[["feature", col_name]].rename(
            columns={col_name: f"importance_by_{model_name.upper()}_SHAP"}
        )

    # normalize importance values for each model
    rf_table = normalize_importance(shap_importance_rf, "rf")
    xgb_table = normalize_importance(shap_importance_xgb, "xgb")
    ada_table = normalize_importance(shap_importance_ada, "ada")
    mlp_table = normalize_importance(shap_importance_mlp, "mlp")
    tabpfn_table = normalize_importance(shap_importance_tabpfn, "tabpfn")

    # merge all tables on the "feature" column
    merged_table = rf_table
    for table in [xgb_table, ada_table, mlp_table, tabpfn_table]:
        merged_table = pd.merge(merged_table, table, on="feature", how="outer")

    # compute the mean importance across all models
    importance_columns = [
        "importance_by_RF_SHAP",
        "importance_by_XGB_SHAP",
        "importance_by_ADA_SHAP",
        "importance_by_MLP_SHAP",
        "importance_by_TABPFN_SHAP",
    ]
    merged_table["mean_importance"] = merged_table[importance_columns].mean(axis=1)

    merged_table = merged_table.sort_values(
        by="mean_importance", ascending=False
    ).reset_index(drop=True)

    merged_table.set_index("feature", inplace=True)
    merged_table = merged_table.astype(float)

    return merged_table


def calculate_percentage(value, total):
    """
    Calculates the percentage of a value out of a total.

    Args:
        value: The part value.
        total: The total value.

    Returns:
        The percentage value, or an error message if total is 0.
    """
    if total == 0:
        return "Total cannot be zero"
    percentage = (value / total) * 100
    return percentage


def false_negatives_exclusion_criteria_counter(false_negatives):
    """
    Prints various exclusion criteria statistics for false negative cases.

    Args:
        false_negatives: Subset of data labeled as false negatives.
    """
    print(
        f"Reported another complaint: '{false_negatives['suggestions_questions'].values[2]}'\n"
    )

    print(
        f"{len(false_negatives[false_negatives['takes_hormones_only_for_contracep'] == 1])} use hormonal contraception.\n"
    )

    print(
        f"{len(false_negatives[false_negatives['takes_hormones_for_pain'] == 1])} use hormonal treatments for pain relief.\n"
    )

    print(f"{false_negatives['age_45_54'].sum()} participants are aged 45-54.\n")

    print(f"{int(false_negatives['was_pregnant'].sum())} participants were pregnant.\n")

    print(
        f"{len(false_negatives[false_negatives['takes_presc_painkillers'] == 1])} use prescribed painkillers.\n"
    )


def false_positives_exclusion_criteria_counter(false_positives):
    """
    Prints various exclusion criteria statistics for false positive cases and returns conditions mentioned.

    Args:
        false_positives: Subset of data labeled as false positives.

    Returns:
        A list of all non-"None of the above" conditions reported.
    """
    false_positives_cond_mentioned = [
        condition
        for condition in false_positives["all_conditions_mentioned"]
        if condition != "None of the above"
    ]
    print(
        f"{len(false_positives_cond_mentioned)} participants reported another complaint: {false_positives_cond_mentioned}\n"
    )

    print(
        f"{len(false_positives[false_positives['takes_hormones_only_for_contracep'] == 1])} use hormonal contraception.\n"
    )

    print(
        f"{len(false_positives[false_positives['takes_hormones_for_pain'] == 1])} use hormonal treatments for pain relief.\n"
    )

    print(
        f"{len(false_positives[false_positives['takes_presc_painkillers'] == 1])} use prescribed painkillers.\n"
    )

    return false_positives_cond_mentioned


def prep_data_for_analysis(df_endo):
    """
    Prepares and formats endometriosis survey data for analysis.

    Args:
        df_endo: Raw survey dataset with features related to symptoms and conditions.

    Returns:
        - DataFrame with endometriosis cases
        - DataFrame without endometriosis cases
        - Frequency mapping dictionary
        - Dictionaries for pelvic, back pain, and headache durations
    """

    # age group categorization
    conditions = [
        df_endo["age_under_18"] == 1,
        df_endo["age_18_24"] == 1,
        df_endo["age_25_34"] == 1,
        df_endo["age_35_44"] == 1,
        df_endo["age_45_54"] == 1,
    ]
    choices = ["Under 18", "18-24", "25-34", "35-44", "45-54"]

    df_endo["age_group"] = np.select(conditions, choices, default="Unknown")

    # helper for BMI categorization
    def bmi_category(bmi):
        """
        Returns the BMI category based on the provided BMI value.

        Used only for the purpose of the data analysis.
        For training and prediction, the BMI value is used directly and not its category.

        Args:
            The BMI value.
        """
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Healthy Weight"
        elif 25 <= bmi < 30:
            return "Overweight"
        elif 30 <= bmi < 35:
            return "Class 1 Obesity"
        elif 35 <= bmi < 40:
            return "Class 2 Obesity"
        else:
            return "Class 3 Obesity"

    # mapping integers to frequency strings
    pain_frequency_map = {
        0: "Never",
        1: "Rarely",
        2: "Sometimes",
        3: "Often",
        4: "Always",
    }

    # duration categories for different types of pain
    PELVIC_PAIN_DURATION_CATEGORIES = {
        0: "I do not experience pelvic pain during my period",
        1: "Less than 1 day",
        2: "1-2 days",
        3: "3–4 days",
        4: "5 or more days",
    }
    LOWER_BACK_PAIN_DURATION_CATEGORIES = {
        0: "I do not experience lower pain during my period",
        1: "Less than 1 day",
        2: "1-2 days",
        3: "3–4 days",
        4: "5 or more days",
    }
    HEADACHE_DURATION_CATEGORIES = {
        0: "I do not experience headaches during my period",
        1: "Less than 1 day",
        2: "1-2 days",
        3: "3–4 days",
        4: "5 or more days",
    }

    # replace all categorical features with readable strings
    mappings = [
        "pelvic_pain_frequency_between_periods",
        "lower_back_pain_frequency_between_periods",
        "pain_in_legs_hips_during_period",
        "pain_in_legs_hips_between_periods",
        "pain_impact_work",
        "bloating_during_period",
        "bloating_between_periods",
        "fatigue_during_period",
        "fatigue_between_periods",
        "experiences_mood_swings",
        "unable_to_cope_with_pain",
        "headache_frequency_between_periods",
        "large_blood_clots_frequency",
        "heavy_bleeding_frequency",
        "night_time_changes",
        "bleeding_impact_social_events",
        "bleeding_impact_home_jobs",
        "bleeding_impact_work",
        "bleeding_impact_physical_activity",
        "pain_impact_social_events",
        "pain_impact_home_jobs",
        "pain_impact_work",
        "pain_impact_physical_activity",
        "pain_impact_appetite",
        "pain_impact_sleep",
    ]

    for col in mappings:
        df_endo[col] = df_endo[col].replace(pain_frequency_map)

    # replace specific columns with duration mappings
    df_endo["pelvic_pain_days_during_period"] = df_endo[
        "pelvic_pain_days_during_period"
    ].replace(PELVIC_PAIN_DURATION_CATEGORIES)
    df_endo["lower_back_pain_days_during_period"] = df_endo[
        "lower_back_pain_days_during_period"
    ].replace(LOWER_BACK_PAIN_DURATION_CATEGORIES)
    df_endo["headache_days_during_period"] = df_endo[
        "headache_days_during_period"
    ].replace(HEADACHE_DURATION_CATEGORIES)

    # assign BMI categories
    df_endo["bmi_category"] = df_endo["BMI"].map(bmi_category)

    # filter into diagnosed and not diagnosed
    endo_df = df_endo[df_endo["has_endometriosis"] == 1]
    no_endo_df = df_endo[df_endo["has_endometriosis"] == 0]

    return (
        endo_df,
        no_endo_df,
        pain_frequency_map,
        PELVIC_PAIN_DURATION_CATEGORIES,
        LOWER_BACK_PAIN_DURATION_CATEGORIES,
        HEADACHE_DURATION_CATEGORIES,
    )


def print_stats(endo_df, no_endo_df, text, feature_endo, cat=None):
    """
    Prints feature distribution between endometriosis and non-endometriosis cases.

    Args:
        endo_df: Participants diagnosed with endometriosis.
        no_endo_df: Participants not diagnosed with endometriosis.
        text: Descriptive text to display before the numbers.
        feature_endo: Column name of the feature to analyze.
        cat (optional): Value/category to filter the feature by.
    """
    if feature_endo in [
        "pelvic_pain_average",
        "pelvic_pain_worst",
        "lower_back_pain_average",
        "lower_back_pain_worst",
        "headache_average",
        "headache_worst",
    ]:
        feature_with_endo = (endo_df[feature_endo] == int(cat)).sum()
        feature_without_endo = (no_endo_df[feature_endo] == int(cat)).sum()
    else:
        feature_with_endo = (
            endo_df[feature_endo].sum()
            if not cat
            else (endo_df[feature_endo] == cat).sum()
        )
        feature_without_endo = (
            no_endo_df[feature_endo].sum()
            if not cat
            else (no_endo_df[feature_endo] == cat).sum()
        )

    endo_cases_num = endo_df.shape[0]
    no_endo_cases_num = no_endo_df.shape[0]

    print(
        f"{text} with endometriosis: {feature_with_endo} "
        f"({calculate_percentage(feature_with_endo, endo_cases_num):.1f}%), "
        f"without endometriosis: {feature_without_endo} "
        f"({calculate_percentage(feature_without_endo, no_endo_cases_num):.1f}%)."
    )
