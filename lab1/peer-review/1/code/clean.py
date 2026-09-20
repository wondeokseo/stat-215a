import numpy as np
import pandas as pd


def clean_ams(df, gcs_total = True):
    """
    Cleans the AMS column in the DataFrame based on the specified conditions.
    
        If GCSTotal is less than 15 or any of the AMS-related columns (AMSAgitated, AMSSleep, AMSSlow, AMSRepeat, AMSOth) are equal to 1, then AMS is set to 1.
        If GCSTotal is equal to 15 and any of the AMS-related columns are equal to 1, then AMS is also set to 1.
        If GCSTotal is equal to 15 and all of the AMS-related columns are equal to 0, then AMS is set to 0.
        If GCSTotal is less than 15 and any of the AMS-related columns are equal to 92, then AMS is set to 1.

        Allows for pertubations of GCS score. Are we using the total? Or the sum of the three components? Default is True, which uses the total.

    
        Parameters:
        df (pd.DataFrame): The input DataFrame.
        gcs_total (bool): Whether to use the GCSTotal column or the sum of the three GCS components. Default is True.
    
        Returns:
        pd.DataFrame: The DataFrame with the cleaned AMS column.

    """

    df_cleaned = df.copy()

    if gcs_total == True:
        df_cleaned['AMS'] = np.where(
            (df_cleaned['GCSTotal'] < 15) |
            (df_cleaned[['AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth']].eq(1).any(axis=1)),
            1,
            np.where(
                (df_cleaned['GCSTotal'] == 15) &
            (df_cleaned[['AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth']].eq(0).any(axis=1)),
            0,
            df_cleaned['AMS']

                )
            )
    else:
        gcs_total = df_cleaned['GCSEye'] + df_cleaned['GCSVerbal'] + df_cleaned['GCSMotor']
        df_cleaned['AMS'] = np.where(
                    (gcs_total < 15) |
                    (df_cleaned[['AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth']].eq(1).any(axis=1)),
                    1,
                    np.where(
                        (gcs_total == 15) &
                    (df_cleaned[['AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth']].eq(0).any(axis=1)),
                    0,
                    df_cleaned['AMS']
        
                        )
                    )

    df_cleaned['AMS'] = df_cleaned['AMS'].replace({1: 'Yes', 0: 'No'})

    df_cleaned['AMS'] = df_cleaned['AMS'].replace(r'^\s*$', np.nan, regex=True)

    return df_cleaned


def clean_palpable(df):
    """
    Cleans the SFxPalp column in the DataFrame based on the specified conditions.

        If SFxPalpDepress is 1 or 0, SFxPalp is set to 1.
        SFxPpalp is then renamed to SFxPalpable.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned SFxPalp column.

    """

    df_cleaned = df.copy()

    df_cleaned['SFxPalp'] = np.where(
        (df_cleaned['SFxPalpDepress'] == 1) | (df_cleaned['SFxPalpDepress'] == 0),
        1,
        df_cleaned['SFxPalp']
    )

    df_cleaned = df_cleaned.rename(columns={'SFxPalp': 'SFxPalpable'})

    df_cleaned['SFxPalpable'] = df_cleaned['SFxPalpable'].replace({1: 'Yes', 0: 'No'})

    df_cleaned['SFxPalpable'] = df_cleaned['SFxPalpable'].replace(r'^\s*$', np.nan, regex=True)

    return df_cleaned


def clean_basillar(df):
    """
    
    Cleans the SFxBas column in the DataFrame based on the specified conditions.

        If any of the SFxBas-related columns (SFxBasHem, SFxBasOto, SFxBasPer, SFxBasRet, SFxBasRhi) are equal to 1, then SFxBas is set to 1.
        If all of the SFxBas-related columns are equal to 0, then SFxBas is set to 0.
        SFxBas is renamed to SFxBasillar.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned SFxBas column.

    """

    df_cleaned = df.copy()

    df_cleaned['SFxBas'] = np.where(
        (df_cleaned[['SFxBasHem', 'SFxBasOto', 'SFxBasPer', 'SFxBasRet', 'SFxBasRhi']].eq(1).any(axis=1)),
        1,
        np.where(
            (df_cleaned[['SFxBasHem', 'SFxBasOto', 'SFxBasPer', 'SFxBasRet', 'SFxBasRhi']].eq(0).all(axis=1)),
            0,
            df_cleaned['SFxBas']
        )
    )

    df_cleaned = df_cleaned.rename(columns={'SFxBas': 'SFxBasillar'})

    df_cleaned['SFxBasillar'] = df_cleaned['SFxBasillar'].replace({1: 'Yes', 0: 'No'})

    df_cleaned['SFxBasillar'] = df_cleaned['SFxBasillar'].replace(r'^\s*$', np.nan, regex=True)

    return df_cleaned


def clean_hematoma(df):
    """
    Cleans the Hema column in the DataFrame based on the specified conditions.

        If HemaLoc or HemaSize is 1, then Hema is set to 1.
        If HemaLoc or HemaSize is 92, then Hema is unchanged. This is because we cannot determine whether Hema should be 0 or 1 if the subcategory is missing.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned Hema column.

    """

    df_cleaned = df.copy()

    df_cleaned['Hema'] = np.where(
        (df_cleaned[['HemaLoc', 'HemaSize']].eq(1).any(axis=1)),
        1,
        df_cleaned['Hema']
    )

    df_cleaned['Hema'] = df_cleaned['Hema'].replace({1: 'Yes', 0: 'No'})

    df_cleaned = df_cleaned.rename(columns={'Hema': 'Hematoma'})

    df_cleaned['Hematoma'] = df_cleaned['Hematoma'].replace(r'^\s*$', np.nan, regex=True)

    return df_cleaned


def clean_neuro_deficit(df):
    """
    Cleans the NeuroD column in the DataFrame based on the specified conditions.

        If any of the NeuroD* columns are equal to 1, then NeuroD is set to 1.
        If all of the NeuroD* columns are equal to 0, then NeuroD is set to 0.
        If all of the NeuroD* columns are equal to 92, then NeuroD is unchanged. This is because we cannot determine whether NeuroD should be 0 or 1 if all of the subcategories are missing.
        If NeuroD = 1, they should be removed from the dataset. We do not want this confounding our analysis.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned NeuroD column.

    """

    df_cleaned = df.copy()

    df_cleaned['NeuroD'] = np.where(
        (df_cleaned[['NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth']].eq(1).any(axis=1)),
        1,
        np.where(
            (df_cleaned[['NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth']].eq(0).all(axis=1)),
            0,
            df_cleaned['NeuroD']
        )
    )

    df_cleaned = df_cleaned[df_cleaned['NeuroD'] != 1]

    return df_cleaned


def clean_outcome(df):
    """

    Cleans the PosIntFinal column in the DataFrame based on the specified conditions.

        If DeathTBI, Neurosurgery, Intub24Head, or HospHead is equal to 1, then PosIntFinal is set to 1.
        If DeathTBI, Neurosurgery, Intub24Head, and HospHead are all equal to 0, then PosIntFinal is set to 0.
        If any of DeathTBI, Neurosurgery, Intub24Head, or HospHead are missing (92), then PosIntFinal is unchanged. This is because we cannot determine whether PosIntFinal should be 0 or 1 if any of the subcategories are missing.
        If after all changes are made PosIntFinal is missing, the patient should be excluded from the analysis. Since this is our outcome, we meed all icluded patients to have an outcome.
        Variable is renamed to ciTBI for clarity. This is the variable we will be using in our analysis.
        
    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned PosIntFinal column.

    """

    df_cleaned = df.copy()

    df_cleaned['HospHead'] = np.where(
        (df_cleaned['HospHeadPosCT'] == 1),
        1,
        df_cleaned['HospHead']
    )  

    df_cleaned['PosIntFinal'] = np.where(
            (df_cleaned[['DeathTBI', 'Neurosurgery', 'Intub24Head', 'HospHead']].eq(1).any(axis=1)),
            1,
            np.where(
                (df_cleaned[['DeathTBI', 'Neurosurgery', 'Intub24Head', 'HospHead']].eq(0).all(axis=1)),
                0,
                df_cleaned['PosIntFinal']
            )
        )

    df_cleaned = df_cleaned.dropna(subset=['PosIntFinal'])

    df_cleaned = df_cleaned.rename(columns={'PosIntFinal': 'ciTBI'})

    df_cleaned['ciTBI'] = df_cleaned['ciTBI'].replace({1: 'Yes', 0: 'No'})
    
    return df_cleaned


def clean_vomit(df):
    """ 

    Cleans the Vomit column in the DataFrame based on the specified conditions. 

        If VomitNbr is greater than 0, then Vomit is set to 1, excluding value of 92 since it denotes missingness. 
        If VomitStart is greater than 0, then Vomit is set to 1, excluding value of 92 since it denotes missingness.
        If VomitLast is greater than 0, then Vomit is set to 1, excluding value of 92 since it denotes missingness.
        Vomit = 0 should be converted to a categorical variable "No" and Vomit = 1 should be converted to a categorical variable "Yes".
        If Vomit continues to be missing (empty), they are kept in the dataset as these patients are still meanignful to the analysis. 
        Ensure missing values are np.nans.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned Vomit column.

    """

    df_cleaned = df.copy()

    df_cleaned['Vomit'] = np.where(
        (df_cleaned['VomitNbr'] > 0) & (df_cleaned['VomitNbr'] != 92),
        1,
        np.where(
            (df_cleaned['VomitStart'] > 0) & (df_cleaned['VomitStart'] != 92),
            1,
            np.where(
                (df_cleaned['VomitLast'] > 0) & (df_cleaned['VomitLast'] != 92),
                1,
                df_cleaned['Vomit']
            )
        )
    )

    df_cleaned['Vomit'] = df_cleaned['Vomit'].replace({0: 'No', 1: 'Yes'})

    df_cleaned['Vomit'] = df_cleaned['Vomit'].replace(r'^\s*$', np.nan, regex=True)

    return df_cleaned


def clean_headache(df):
    """ 

    Cleans the HASeverity column in the DataFrame based on the specified conditions. 

        If HASeverity is greater than 0, then HA_verb is set to 1, excluding value of 92 since it denotes missingness. 
        If HAStart is greater than 0, then HA_verb is set to 1, excluding value of 92 since it denotes missingness.
        IF HA_verb is 91, this is meaninful and should be kept. 91 means they are nonverbal or preverbal and cannot report a headache.
        Additionally, there is no logical way to impute headache if it is missing and does not meet the criteria.
        HA_verb = 0 should be converted to a categorical variable "No", HA_verb = 1 should be converted to a categorical variable "Yes", and HA_verb = 91 should be converted to a categorical variable "Nonverbal/Preverbal".
        If HA_verb continues to be missing (empty), they are kept in the dataset as these patients are still meanignful to the analysis. 
        Ensure missing values are np.nans.
        Variable is renamed to Headache.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned HASeverity column.

    """

    df_cleaned = df.copy()

    df_cleaned['HA_verb'] = np.where(
        (df_cleaned['HASeverity'] > 0) & (df_cleaned['HASeverity'] != 92),
        1,
        np.where(
            (df_cleaned['HAStart'] > 0) & (df_cleaned['HAStart'] != 92),
            1,
            df_cleaned['HA_verb']
        )
    )

    df_cleaned['HA_verb'] = df_cleaned['HA_verb'].replace({0: 'No', 1: 'Yes', 91: 'Nonverbal/Preverbal'})

    df_cleaned['HA_verb'] = df_cleaned['HA_verb'].replace(r'^\s*$', np.nan, regex=True)

    df_cleaned['HA_verb'] = df_cleaned['HA_verb'].replace({91: np.nan})

    df_cleaned = df_cleaned.rename(columns={'HA_verb': 'Headache'})

    return df_cleaned


def clean_loc(df, suspected = True):
    """ 
    Cleans the LOCSeparate column in the DataFrame based on the specified conditions.
        
        If suspected is True, then any patients with LOCSeparate = 2 (suspected) and LocLen in [1, 2, 3, 4] will be set to LOCSeparate = 1 (Yes).
        If suspected is False, then no changes will be made to the LOCSeparate column.
        LOCSeparate is renamed to LOC.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    suspected (bool): Whether to clean suspected LOC cases based on LocLen. Default is True

    Returns:
    pd.DataFrame: The DataFrame with the cleaned LOCSeparate column.

    """

    df_cleaned = df.copy()

    if suspected:
        valid_loc_durations = [1, 2, 3, 4]
        cond = df_cleaned['LocLen'].isin(valid_loc_durations)
        mask = (df_cleaned['LOCSeparate'] == 2) & cond
        df_cleaned.loc[mask, 'LOCSeparate'] = 1

    df_cleaned['LOCSeparate'] = df_cleaned['LOCSeparate'].replace(r'^\s*$', np.nan, regex=True)

    df_cleaned = df_cleaned.rename(columns={'LOCSeparate': 'LOC'})

    return df_cleaned


def clean_injury_severity(df, new_severity = True):
    """
    Cleans the High_impact_InjSev column in the DataFrame based on the specified conditions.

        If new_severity is True, then any patients with InjuryMech = 2 (Pedestrian) and missing High_impact_InjSev will be set to High_impact_InjSev = 3 (High).
        If new_severity is False, then no changes will be made to the High_impact_InjSev column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    new_severity (bool): Whether to clean missing severity labels for InjuryMech = 2. Default is True.

    Returns:
    pd.DataFrame: The DataFrame with the cleaned High_impact_InjSev column.
    """

    df_cleaned = df.copy()

    if new_severity:
        missing_severity_mask = df_cleaned['InjuryMech'].eq(2) & df_cleaned['High_impact_InjSev'].isna()
        df_cleaned.loc[missing_severity_mask, 'High_impact_InjSev'] = 3

    df_cleaned['High_impact_InjSev'] = df_cleaned['High_impact_InjSev'].replace(r'^\s*$', np.nan, regex=True)

    df_cleaned = df_cleaned.rename(columns={'High_impact_InjSev': 'InjurySeverity'})

    return df_cleaned


def clean(df, gcs_total=True, suspected=True, new_severity=True):
    """
    Cleans the DataFrame by applying all the cleaning functions in sequence.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    gcs_total (bool): Whether to use the GCSTotal column or the sum of the three GCS components. Default is True.
    suspected (bool): Whether to clean suspected LOC cases based on LocLen. Default is True.
    new_severity (bool): Whether to clean missing severity labels for InjuryMech = 2. Default is True.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """

    df_cleaned = clean_ams(df, gcs_total)
    df_cleaned = clean_palpable(df_cleaned)
    df_cleaned = clean_basillar(df_cleaned)
    df_cleaned = clean_hematoma(df_cleaned)
    df_cleaned = clean_neuro_deficit(df_cleaned)
    df_cleaned = clean_outcome(df_cleaned)
    df_cleaned = clean_vomit(df_cleaned)
    df_cleaned = clean_headache(df_cleaned)
    df_cleaned = clean_loc(df_cleaned, suspected)
    df_cleaned = clean_injury_severity(df_cleaned, new_severity)

    df_cleaned = df_cleaned[['AgeTwoPlus', 'InjuryMech', 'InjurySeverity', 'LOC', 'Headache', 'ActNorm', 'Vomit', 'AMS', 'SFxPalpable', 'SFxBasillar', 'Hematoma', 'Gender', 'ciTBI', 'Race', 'CTDone']]

    return df_cleaned
