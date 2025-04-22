"""
This module supports preparation of the dataset for model development.
"""

import pandas as pd
import re
from sklearn.impute import SimpleImputer


def rename_columns(df):
    """
    Renames columns in the provided DataFrame for consistency and readability.

    Args:
        df: The DataFrame to rename columns for.
    """
    return df.rename(
        columns={
            "How old are you?": "age_group",
            "What is your ethnicity? (Select all that apply)": "ethnicity",
            "Country of origin:": "country",
            "How tall are you?": "height",
            "What is your current weight?": "weight",
            "Are you currently pregnant?": "is_pregnant",
            "Have you ever been pregnant (confirmed by a positive pregnancy test, including miscarriages, ectopic pregnancies or terminations)?": "was_pregnant",
            "Have you experienced infertility issues?": "experienced_infertility",
            "Have you undergone hysterectomy?": "had_hysterectomy",
            "Have you been diagnosed with thyroid disorders?": "has_thyroid_disorder",
            "In the past 12 months, have you been using hormonal contraceptives?": "use_hormonal_contracep",
            "Have you ever received a diagnosis for any of the following conditions?": "all_conditions_mentioned",
            "In the past 12 months, have you experienced menstruation?": "is_menstruating",
            "Have you been diagnosed with anemia?": "has_anemia",
            "In the past 12 months, how often did you experience pelvic pain? (Select all that apply)": "pelvic_pain_timing",
            "In the past 12 months, how many days on average did you experience pelvic pain during your period?": "pelvic_pain_days_during_period",
            "In the past 12 months, how often did you experience pelvic pain between periods?": "pelvic_pain_frequency_between_periods",
            "Rate your pelvic pain at its worst (considering pain during your period and between periods)": "pelvic_pain_worst",
            "Rate your pelvic pain at its average (considering pain during your period and between periods)": "pelvic_pain_average",
            "In the past 12 months, how often did you experience lower back pain? (Select all that apply)": "lower_back_pain_frequency",
            "In the past 12 months, how many days on average did you experience lower back pain during your period?": "lower_back_pain_days_during_period",
            "In the past 12 months, how often did you experience lower back pain between periods?": "lower_back_pain_frequency_between_periods",
            "Rate your lower back pain at its worst (considering pain during your period and between periods)": "lower_back_pain_worst",
            "Rate your lower back pain at its average (considering pain during your period and between periods)": "lower_back_pain_average",
            "In the past 12 months, how often did you experience headaches? (Select all that apply)": "headache_frequency",
            "In the past 12 months, how many days on average did you experience headaches during your period?": "headache_days_during_period",
            "In the past 12 months, how often did you experience headaches between periods?": "headache_frequency_between_periods",
            "Rate your headaches at its worst (considering pain during your period and between periods)": "headache_worst",
            "Rate your headaches at its average (considering pain during your period and between periods)": "headache_average",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Abdominal pressure DURING your period]": "abdominal_pressure_during_period",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Abdominal pressure BETWEEN your periods]": "abdominal_pressure_between_periods",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Breast discomfort DURING your period]": "breast_discomfort_during_period",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Breast discomfort BETWEEN your periods]": "breast_discomfort_between_periods",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Bloating DURING your period]": "bloating_during_period",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Bloating BETWEEN your periods]": "bloating_between_periods",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Pain in legs/hips DURING your period]": "pain_in_legs_hips_during_period",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Pain in legs/hips BETWEEN your periods]": "pain_in_legs_hips_between_periods",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Fatigue DURING your period]": "fatigue_during_period",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Fatigue BETWEEN your periods]": "fatigue_between_periods",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Nausea DURING your period]": "nausea_during_period",
            "In the past 12 months, how often did you experience ... (choose from Never to Always) [Nausea BETWEEN your periods]": "nausea_between_periods",
            "Do you take medication to relieve the menstrual pain?": "takes_medication_for_pain",
            "Do you experience pain during sexual intercourse? (Select all that apply)": "pain_during_intercourse",
            "Where exactly do you experience pain during sexual intercourse? (Select all that apply)": "pain_locations_intercourse",
            "Do you ... [Have regular bowel movements (from once a day to 3-4 times a week)?]": "regular_bowel_movements",
            "Do you ... [Experience painful bowel movements?]": "painful_bowel_movements",
            "Do you ... [Experience difficulty controlling urination?]": "difficulty_controlling_urination",
            "Do you ... [Experience frequent urination?]": "frequent_urination",
            "Do you ... [Experience pain or discomfort when urinating?]": "pain_during_urination",
            "How long is your typical menstrual cycle?": "menstrual_cycle_length",
            "How many days do you typically bleed during your period?": "period_duration",
            "Does the duration of your menstrual bleeding change from month to month?": "bleeding_duration_changes",
            "How often do you notice large blood clots (about 1 inch or more)?": "large_blood_clots_frequency",
            "Do you experience spotting or bleeding between periods?": "spotting_between_periods",
            "How often do you experience heavy bleeding that requires you to change tampons or pads every hour for several hours in the row?": "heavy_bleeding_frequency",
            "Do you need to get up in the night in order to change sanitary products (in order to prevent leakage)?": "night_time_changes",
            "How would you describe the heaviness of your menstrual bleeding?": "bleeding_heaviness",
            "Is there a family history of gynecological conditions [Endometriosis]": "family_history_endometriosis",
            "Is there a family history of gynecological conditions [Fibroids]": "family_history_fibroids",
            "Is there a family history of gynecological conditions [Polycystic ovary syndrome (PCOS)]": "family_history_pcos",
            "Is there a family history of gynecological conditions [Infertility]": "family_history_infertility",
            "Is there a family history of gynecological conditions [Heavy bleeding]": "family_history_heavy_bleeding",
            "Is there a family history of gynecological conditions [Pelvic pain symptoms]": "family_history_pelvic_pain",
            "In the past 12 months, how often, because of (heavy) bleeding during your period, have you: [Been unable to go to social events?]": "bleeding_impact_social_events",
            "In the past 12 months, how often, because of (heavy) bleeding during your period, have you: [Been unable to do jobs around the home?]": "bleeding_impact_home_jobs",
            "In the past 12 months, how often, because of (heavy) bleeding during your period, have you: [Missed work or university?]": "bleeding_impact_work",
            "In the past 12 months, how often, because of (heavy) bleeding during your period, have you: [Reduced or stopped physical activity?]": "bleeding_impact_physical_activity",
            "In the past 12 months, how often, because of pain, have you: [Been unable to go to social events?]": "pain_impact_social_events",
            "In the past 12 months, how often, because of pain, have you: [Been unable to do jobs around the home?]": "pain_impact_home_jobs",
            "In the past 12 months, how often, because of pain, have you: [Missed work or university?]": "pain_impact_work",
            "In the past 12 months, how often, because of pain, have you: [Reduced or stopped physical activity?]": "pain_impact_physical_activity",
            "In the past 12 months, how often, because of pain, have you: [Lost appetite and/or been unable to eat]": "pain_impact_appetite",
            "In the past 12 months, how often, because of pain, have you: [Been unable to sleep properly]": "pain_impact_sleep",
            "Do you experience mood swings?": "experiences_mood_swings",
            "Have you felt that you are unable to cope with pain?": "unable_to_cope_with_pain",
            "Suggestions & Questions": "suggestions_questions",
        }
    )


# Symptom mappings
MULT_CHOICE_SYMPTOMS = {
    # Age groups
    "age_under_18": ("age_group", "Under 18"),
    "age_18_24": ("age_group", "18–24"),
    "age_25_34": ("age_group", "25–34"),
    "age_35_44": ("age_group", "35–44"),
    "age_45_54": ("age_group", "45–54"),
    "age_55_64": ("age_group", "55–64"),
    # Timing of pelvic pain
    "pelvic_pain_during_period": ("pelvic_pain_timing", "during my period"),
    "pelvic_pain_before_period": (
        "pelvic_pain_timing",
        "a few days (1-2) before my period starts",
    ),
    "pelvic_pain_after_period": (
        "pelvic_pain_timing",
        "a few days (1-2) after my period ends",
    ),
    "pelvic_pain_between_periods": ("pelvic_pain_timing", "between my periods"),
    # Frequency of lower back pain
    "lower_back_pain_during_period": ("lower_back_pain_frequency", "during my period"),
    "lower_back_pain_before_period": (
        "lower_back_pain_frequency",
        "a few days (1-2) before my period starts",
    ),
    "lower_back_pain_after_period": (
        "lower_back_pain_frequency",
        "a few days (1-2) after my period ends",
    ),
    "lower_back_pain_between_periods": (
        "lower_back_pain_frequency",
        "between my periods",
    ),
    # Frequency of headache
    "headache_during_period": ("headache_frequency", "during my period"),
    "headache_before_period": (
        "headache_frequency",
        "a few days (1-2) before my period starts",
    ),
    "headache_after_period": (
        "headache_frequency",
        "a few days (1-2) after my period ends",
    ),
    "headache_between_periods": ("headache_frequency", "between my periods"),
    # Duration of menstrual period
    "1_2days_period_duration": ("period_duration", "1–2 days"),
    "3_4days_period_duration": ("period_duration", "3–4 days"),
    "5_6days_period_duration": ("period_duration", "5–6 days"),
    "7plus_days_period_duration": ("period_duration", "7+ days"),
    # Menstrual cycle length
    "less24d_cycle_length": ("menstrual_cycle_length", "Less than 24 days"),
    "24_31d_cycle_length": ("menstrual_cycle_length", "24–31 days"),
    "32_38d_cycle_length": ("menstrual_cycle_length", "32–38 days"),
    "39_50d_cycle_length": ("menstrual_cycle_length", "39–50 days"),
    "more51d_cycle_length": ("menstrual_cycle_length", "More than 51 days"),
    "too_irregular_cycle_length": (
        "menstrual_cycle_length",
        "Too irregular to estimate",
    ),
    # Medications taken for pain relief
    "takes_over_cntr_pills": ("takes_medication_for_pain", "over-the-counter"),
    "takes_hormones_for_pain": ("takes_medication_for_pain", "hormones"),
    "takes_presc_painkillers": ("takes_medication_for_pain", "prescribed"),
    "no_meds": ("takes_medication_for_pain", "No"),
    # Pain during intercourse
    "not_sex_active": ("pain_during_intercourse", "I am not sexually active"),
    "no_sex_pain": ("pain_during_intercourse", "No pain during intercourse"),
    "pain_after_sex": ("pain_during_intercourse", "Pain after intercourse"),
    # Locations of pain during intercourse
    "pelvic_pain_during_intercourse": ("pain_locations_intercourse", "Pelvic pain"),
    "abdominal_pain_during_intercourse": (
        "pain_locations_intercourse",
        "Abdominal pain",
    ),
    "deep_vaginal_pain_during_intercourse": (
        "pain_locations_intercourse",
        "Deep vaginal pain",
    ),
}

BINARY_FEATURES = [
    # Bowel-related symptoms
    "regular_bowel_movements",
    "painful_bowel_movements",
    "difficulty_controlling_urination",
    # Urinary-related symptoms
    "frequent_urination",
    "pain_during_urination",
    # Menstrual and reproductive symptoms
    "bleeding_duration_changes",
    "spotting_between_periods",
    "was_pregnant",
    "experienced_infertility",
    # Surgical and health history
    "had_hysterectomy",
    "has_thyroid_disorder",
    "has_anemia",
    "use_hormonal_contracep",
]

FAMILY_HISTORY = [
    # Family history related to reproductive health
    "family_history_endometriosis",
    "family_history_fibroids",
    "family_history_pcos",
    "family_history_infertility",
    "family_history_heavy_bleeding",
    "family_history_pelvic_pain",
]

# Possible responses indicating the frequency of an event or symptom
RELATIVE_FREQUENCY_CATEGORIES = ["Never", "Rarely", "Sometimes", "Often", "Always"]

# Possible responses indicating the frequency of blood clots during menstruation
BLOOD_CLOTS_CATEGORIES = ["Never", "Rarely", "Sometimes", "Often"]

# Possible responses indicating the heaviness of bleeding during menstruation
BLEEDING_HEAVINESS_CATEGORIES = ["Light", "Moderate", "Heavy", "Very Heavy"]

# Possible responses indicating the duration of pelvic pain experienced during menstruation.
PELVIC_PAIN_DURATION_CATEGORIES = [
    "I do not experience pelvic pain during my period",
    "Less than 1 day",
    "1-2 days",
    "3–4 days",
    "5 or more days",
]

# Possible responses indicating the duration of lower back pain experienced during menstruation.
LOWER_BACK_PAIN_DURATION_CATEGORIES = [
    "I do not experience lower pain during my period",
    "Less than 1 day",
    "1-2 days",
    "3–4 days",
    "5 or more days",
]

# Possible responses indicating the duration of headaches experienced during menstruation.
HEADACHE_DURATION_CATEGORIES = [
    "I do not experience headaches during my period",
    "Less than 1 day",
    "1-2 days",
    "3–4 days",
    "5 or more days",
]

# Mapping of column names to their corresponding symptom categories
COL_CTG_MAPPING = {
    "pelvic_pain_days_during_period": PELVIC_PAIN_DURATION_CATEGORIES,
    "lower_back_pain_days_during_period": LOWER_BACK_PAIN_DURATION_CATEGORIES,
    "headache_days_during_period": HEADACHE_DURATION_CATEGORIES,
    "pelvic_pain_frequency_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "lower_back_pain_frequency_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "headache_frequency_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "abdominal_pressure_during_period": RELATIVE_FREQUENCY_CATEGORIES,
    "abdominal_pressure_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "breast_discomfort_during_period": RELATIVE_FREQUENCY_CATEGORIES,
    "breast_discomfort_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "bloating_during_period": RELATIVE_FREQUENCY_CATEGORIES,
    "bloating_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "pain_in_legs_hips_during_period": RELATIVE_FREQUENCY_CATEGORIES,
    "pain_in_legs_hips_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "fatigue_during_period": RELATIVE_FREQUENCY_CATEGORIES,
    "fatigue_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "nausea_during_period": RELATIVE_FREQUENCY_CATEGORIES,
    "nausea_between_periods": RELATIVE_FREQUENCY_CATEGORIES,
    "bleeding_impact_social_events": RELATIVE_FREQUENCY_CATEGORIES,
    "bleeding_impact_home_jobs": RELATIVE_FREQUENCY_CATEGORIES,
    "bleeding_impact_work": RELATIVE_FREQUENCY_CATEGORIES,
    "bleeding_impact_physical_activity": RELATIVE_FREQUENCY_CATEGORIES,
    "pain_impact_social_events": RELATIVE_FREQUENCY_CATEGORIES,
    "pain_impact_home_jobs": RELATIVE_FREQUENCY_CATEGORIES,
    "pain_impact_work": RELATIVE_FREQUENCY_CATEGORIES,
    "pain_impact_physical_activity": RELATIVE_FREQUENCY_CATEGORIES,
    "pain_impact_appetite": RELATIVE_FREQUENCY_CATEGORIES,
    "pain_impact_sleep": RELATIVE_FREQUENCY_CATEGORIES,
    "experiences_mood_swings": RELATIVE_FREQUENCY_CATEGORIES,
    "unable_to_cope_with_pain": RELATIVE_FREQUENCY_CATEGORIES,
    "large_blood_clots_frequency": BLOOD_CLOTS_CATEGORIES,
    "heavy_bleeding_frequency": RELATIVE_FREQUENCY_CATEGORIES,
    "night_time_changes": RELATIVE_FREQUENCY_CATEGORIES,
    "bleeding_heaviness": BLEEDING_HEAVINESS_CATEGORIES,
}


# Conversion functions
def convert_weight(weight):
    """
    Standardize weight to kilograms.

    Args:
        weight: The weight to convert.
    """
    if not isinstance(weight, str):
        return None
    value = re.findall(r"\d+", weight)
    if not value:
        return None
    value = float(value[0])
    return value * 0.453592 if "lbs" in weight.lower() or value > 100 else value


def convert_height(height):
    """
    Standardize height to centimeters.

    Args:
        height: The height to convert.
    """
    if not isinstance(height, str) or height.lower() in ["nan", "no"]:
        return None
    height = height.strip().replace("’", "'").replace("”", '"')
    match = re.search(r"(\d+\.?\d*)\s*(cm|m|feet|ft|'|in|inches)?", height.lower())
    if match:
        value, unit = float(match.group(1)), match.group(2)
        if unit in ["cm", "centimetres", "centimeters"]:
            return round(value, 2)
        if unit in ["m", "metres"]:
            return round(value * 100, 2)
        if unit in ["in", "inches"]:
            return round(value * 2.54, 2)
        if unit in ["ft", "feet", "'"]:
            return round(value * 30.48, 2)
    return None


# Preprocessing functions
def prep_mult_choice_symptoms(df):
    """
    Prepares the DataFrame by converting multiple choice symptoms into binary features.

    Args:
        df: The DataFrame.
    """
    for new_col, (source_col, condition) in MULT_CHOICE_SYMPTOMS.items():
        df[new_col] = (
            df[source_col].astype(str).str.contains(condition, regex=False).astype(int)
        )
    return df.drop(
        columns=[col for col, _ in MULT_CHOICE_SYMPTOMS.values() if col in df.columns],
        axis=1,
    )


def prep_binary_features(df):
    """
    Prepares the DataFrame binary features.

    Args:
        df: The DataFrame.
    """
    return df.assign(
        **{
            col: df[col].map({"Yes": 1, "No": 0}, na_action="ignore")
            for col in BINARY_FEATURES
        }
    )


def prep_family_history(df):
    """
    Prepares the DataFrame by converting family history features into binary features.

    Args:
        df: The DataFrame.
    """
    return df.assign(
        **{
            col: df[col]
            .astype(str)
            .str.contains("Choose if you know", regex=False)
            .astype(int)
            for col in FAMILY_HISTORY
        }
    )


def prep_mapping(df):
    """
    Maps categorical features to numerical values.

    Args:
        df: The DataFrame.
    """
    return df.replace(
        {
            col: {k: v for v, k in enumerate(categories)}
            for col, categories in COL_CTG_MAPPING.items()
        }
    )


def adjust_scores(df):
    """
    Adjusts the scores for pelvic pain, lower back pain, and headache to start from 0.

    Args:
        df: The DataFrame.
    """
    score_vars = [
        "pelvic_pain_average",
        "lower_back_pain_average",
        "headache_average",
        "pelvic_pain_worst",
        "lower_back_pain_worst",
        "headache_worst",
    ]
    for var in score_vars:
        df[var] = df[var] - 1

    return df


def preprocess_hormal_usage(df):
    """
    Preprocesses the hormonal usage features.

    Creates a new feature that indicates whether the respondent takes hormones only for contraception.

    Args:
        df: The DataFrame.
    """
    df["takes_hormones_only_for_contracep"] = (df["takes_hormones_for_pain"] == 0) & (
        df["use_hormonal_contracep"] == 1
    )
    return df


def preprocess_gyn_data(df):
    """
    Preprocesses the endoemtriosis data.

    Args:
        df: The DataFrame.
    """
    df["height"] = [convert_height(h) for h in df["height"]]
    df["weight"] = [convert_weight(w) for w in df["weight"]]
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    df = prep_mult_choice_symptoms(df)
    df = prep_binary_features(df)
    df = prep_family_history(df)
    df = prep_mapping(df)
    df = adjust_scores(df)
    df = preprocess_hormal_usage(df)
    return df


def impute_features(train_var_set, test_var_set):
    numerical_columns = ["BMI"]

    categorical_columns = [
        col for col in train_var_set.columns if col not in numerical_columns
    ]

    # For binary/categorical features
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    # For numerical features (height, weight, BMI)
    numerical_imputer = SimpleImputer(strategy="mean")

    train_var_set[categorical_columns] = categorical_imputer.fit_transform(
        train_var_set[categorical_columns]
    )
    test_var_set[categorical_columns] = categorical_imputer.transform(
        test_var_set[categorical_columns]
    )

    if numerical_columns[0] in train_var_set.columns:
        train_var_set[numerical_columns] = numerical_imputer.fit_transform(
            train_var_set[numerical_columns]
        )
        test_var_set[numerical_columns] = numerical_imputer.transform(
            test_var_set[numerical_columns]
        )

    return train_var_set, test_var_set
