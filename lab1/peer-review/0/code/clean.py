import pandas as pd
import numpy as np

def clean_data(train_data, val_data, test_data, alternate_impute=False):
    """ Cleans the data: imputes missing values and removing undesired columns
    
    Args:
        train_data (pd.DataFrame): Training dataset
        val_data (pd.DataFrame): Validation dataset
        test_data (pd.DataFrame): Test dataset
        alternate_impute (bool, optional): Boolean variable for toggling alternative imputing methods
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - datasets[0] (pd.DataFrame): Cleaned training dataset
            - datasets[1] (pd.DataFrame): Cleaned validation dataset
            - datasets[2] (pd.DataFrame): Cleaned test dataset
    """
    datasets = [train_data.copy(), val_data.copy(), test_data.copy()]

    # Variables to store training proportions across iterations
    train_overall_vals, train_overall_probs, train_group_props = None, None, None
    proportions = [1, 1]

    for i in range(len(datasets)):
        data = datasets[i]

        if alternate_impute:
            # Impute main symptom variables using the Ind variables
            data = impute_from_ind(data)

            # If the current dataset is the training dataset, store proportions of injury severity to use on all datasets
            if i == 0:
                overall_counts = data["High_impact_InjSev"].dropna().value_counts(normalize=True)
                train_overall_vals = overall_counts.index.values
                train_overall_probs = overall_counts.values

                # Get proportions, grouped by injury mechanism
                train_group_props = (data.dropna(subset=["High_impact_InjSev"]).groupby("InjuryMech")["High_impact_InjSev"].value_counts(normalize=True))

            # If the injury severity is a primary cause for ordering the CT scan, set injury severity to 3
            data.loc[(data["IndMech"] == 1) & (data["High_impact_InjSev"].isna()), "High_impact_InjSev"] = 3

            # Impute all remaining NaNs using saved training proportions
            impute_mask = data["High_impact_InjSev"].isna()
            data.loc[impute_mask, "High_impact_InjSev"] = data[impute_mask].apply(impute_row, args=(train_overall_vals, train_overall_probs, train_group_props), axis=1,)
            
        else:
            # Impute the main symptom variables using the sub symptom variables
            data = impute_from_sub_vars(data)

            mask1 = (data["InjuryMech"] == 2) & (data["AgeTwoPlus"] == 1) & (data["High_impact_InjSev"].isna())
            mask2 = (data["InjuryMech"] == 2) & (data["AgeTwoPlus"] == 2) & (data["High_impact_InjSev"].isna())


            if (i == 0):
                # Get proportions of injury severity = 3.0 for injury mechanism = 2, grouped by age group
                proportions = (pd.crosstab(data[data["InjuryMech"] == 2]["AgeTwoPlus"], data[data["InjuryMech"] == 2]["High_impact_InjSev"], normalize="index")[3.0].tolist())

            p1, p2 = proportions[0], proportions[1]

            if mask1.sum() > 0:
                data.loc[mask1, "High_impact_InjSev"] = np.random.choice([3.0, 2.0], size=mask1.sum(), p=[p1, 1.0 - p1])
            if mask2.sum() > 0:
                data.loc[mask2, "High_impact_InjSev"] = np.random.choice([3.0, 2.0], size=mask2.sum(), p=[p2, 1.0 - p2])


        # Get inverse of ActNorm so that all variables are reflect a positive symptom when variable = 1
        data["acting_abnormally"] = 1 - data["ActNorm"]
        
        # Remove rows: any rows in which the sum of the GCS components do not sum up to the total, missing values for gender, unclear results, and missing values from main variables
        data = remove_GCS_mismatch(data)
        data = data[data["Gender"].notna()]
        data = data[~(data["LOCSeparate"] == 2) & ~(data["SFxPalp"] == 2)]
        data = data.rename(columns={"SFxPalp": "palpable_skull_fracture",
                                    "AgeTwoPlus": "age_group",
                                    "Gender": "gender",
                                    "PosIntFinal": "ci_tbi",
                                    "LOCSeparate": "loss_of_consciousness",
                                    "High_impact_InjSev": "injury_severity",
                                    "AMS": "altered_mental_status", 
                                    "SFxBas": "basilar_skull_fracture", 
                                    "Clav": "clavicle_trauma", 
                                    "NeuroD": "neurological_deficit", 
                                    "Seiz": "seizure", 
                                    "OSI": "other_substantial_injuries", 
                                    "Vomit": "vomiting", 
                                    "Hema": "scalp_hematoma",
                                    "GCSTotal": "gcs_total"})

        columns_to_keep = ["injury_severity", "altered_mental_status", 
                           "basilar_skull_fracture", "clavicle_trauma", 
                           "neurological_deficit", "seizure", 
                           "other_substantial_injuries", "gcs_total",
                           "vomiting", "scalp_hematoma",
                           "palpable_skull_fracture", "age_group",
                           "gender", "ci_tbi",
                           "loss_of_consciousness", "acting_abnormally"]
        
        data = data[columns_to_keep].astype("Int64")

        # Convert values from int to str manually
        data["age_group"] = ["< 2 Years" if val == 1 else ">= 2 Years" for val in data["age_group"]]
        data["gender"] = ["Male" if val == 1 else "Female" for val in data["gender"]]
        data["injury_severity"] = data["injury_severity"].astype(str).replace({"1": "Low", "2": "Medium", "3": "High"})
    
        datasets[i] = data.dropna()

    return datasets[0], datasets[1], datasets[2]

def impute_from_sub_vars(df):
    """ Imputes main symptom variables using sub symptom variables
    Args:
        df (pd.DataFrame): Dataset to impute
        
    Returns:
        df (pd.DataFrame): The imputed dataset
    """
    df = df.copy()
    main_vars = ["AMS", "SFxBas", "Clav", "NeuroD", "Seiz", "OSI", "CTSed", "Vomit", "Hema"]
    
    for main_var in main_vars:
        # Get sub symptom variables (all of which whose names start with the name of the main symptom variable)
        pattern = f"^{main_var}."
        sub_cols = [col for col in df.columns if col != main_var and pd.Series(col).str.contains(pattern).iloc[0]]

        # main symptom variable = 1 if 0 or 1 in sub symptom variables, 0 otherwise
        mask = df[main_var].isna()
        is_one = ((df.loc[mask, sub_cols] != 92) & (df.loc[mask, sub_cols].notna())).any(axis=1)
        is_zero = (df.loc[mask, sub_cols] == 92).any(axis=1)
        
        conditions = [is_one, is_zero]
        outcomes = [1, 0]
        
        df.loc[mask, main_var] = np.select(conditions, outcomes, default=df.loc[mask, main_var])
    
    return df

def impute_from_ind(df):
    """ Imputes main symptom variables using impute variables starting with "Ind"
    Args:
        df (pd.DataFrame): Dataset to impute
        
    Returns:
        df (pd.DataFrame): The imputed dataset
    """
    df = df.copy()

    # Impute main symptom variables with 1 if Ind variable == 1, 0 otherwise
    mask_AMS = df["AMS"].isna()
    df.loc[mask_AMS, "AMS"] = np.where(df.loc[mask_AMS, "IndAMS"] == 1, 1, 0)

    mask_Hema = df["Hema"].isna()
    df.loc[mask_Hema, "Hema"] = np.where(df.loc[mask_Hema, "IndHema"] == 1, 1, 0)

    mask_LOC = df["LOCSeparate"].isna()
    df.loc[mask_LOC, "LOCSeparate"] = np.where(df.loc[mask_LOC, "IndLOC"] == 1, 1, 0)

    mask_skull_fracture = df["SFxPalp"].isna()
    df.loc[mask_skull_fracture, "SFxPalp"] = np.where((df.loc[mask_skull_fracture, "IndClinSFx"] == 1) | (df.loc[mask_skull_fracture, "IndXraySFx"] == 1), 1, 0)

    mask_ActNorm = df["ActNorm"].isna()
    df.loc[mask_ActNorm, "ActNorm"] = np.where(df.loc[mask_ActNorm, "IndRqstParent"] == 1, 0, 1)

    return df

def impute_row(row, overall_vals, overall_probs, group_props):
    """ 
    
    Imputes missing injury severity based on injury mechanism grouping or overall sample proportions.
    
    Args:
        row (pd.Series): Current row being imputed
        overall_vals (array-like): Available values for imputing (1, 2, or 3)
        overall_probs (array-like): Proportions of each element in overall_vals from non-missing entries in the training data
        group_props (pd.DataFrame): Proportions of injury severity scores stratified by injury mechanism from non-missing entries in the training data 
        
    Returns:
        (int) Imputed value
    """
    mech = row["InjuryMech"]
    
    # Impute missing value with values from proportions obtained from the whole sample using probabilities of each potential outcome (ex P(High_impact_InjSev = 1))
    if pd.isna(mech):
        return np.random.choice(overall_vals, p=overall_probs)

    # Impute missing value with values from proportions obtained from given injury mechanism using probabilities of each potential outcome (ex P(High_impact_InjSev = 1 | InjuryMech = 1))
    mech_props = group_props.loc[mech]
    return np.random.choice(mech_props.index.values, p=mech_props.values)

def remove_GCS_mismatch(df):
    """ 
    
    Removes rows in which the sum of the components of the GCS do not match GCS_total.
    
    Args:
        df (pd.DataFrame): The dataframe being imputed
    
    Returns:
        df (pd.DataFrame): Imputed dataframe
    """
    mask_gcs_not_na = df["GCSEye"].notna() & df["GCSVerbal"].notna() & df["GCSMotor"].notna()
    criteria_mismatch_sum = df["GCSTotal"] != (df["GCSEye"] + df["GCSVerbal"] + df["GCSMotor"])

    df = df[~(mask_gcs_not_na & criteria_mismatch_sum)]

    return df