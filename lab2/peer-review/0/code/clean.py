"""Clean and preprocess the dialect survey data."""

import geopandas as gpd
import pandas as pd

COLUMN_RENAMES = {
    "ID": "respondent_id",
    "CITY": "city",
    "STATE": "state",
    "ZIP": "zip_code",
    "lat": "latitude",
    "long": "longitude",
}

RESPONDENT_COLUMNS = [
    "respondent_id",
    "city",
    "state",
    "zip_code",
    "latitude",
    "longitude",
    "location_conflict",
]

MAX_STATE_DISTANCE_KM = 100

DISTANCE_CRS = "EPSG:5070"


def get_question_columns(data):
    """Return Q-prefixed survey column names in their original order.

    Args:
        data (pandas.DataFrame): Raw or cleaned survey data with string
            column names, such as Q050 or Q105 for survey questions.

    Returns:
        list[str]: All column names starting with Q, in their original order.
    """
    return [column for column in data.columns if column.startswith("Q")]

def clean_data(data, question_data, state_shapes):
    """Label survey responses and flag inconsistent geographic information.

    Args:
        data (pandas.DataFrame): Raw lingData data with ID, CITY, STATE, ZIP,
            lat, long, and Q-prefixed responses. Answer codes are 1-based;
            0 means no response. Coordinates are WGS84 degrees.
        question_data (dict): Tables returned by pyreadr.read_r. Each ans.N
            table supplies answer text in the survey's answer-code order.
        state_shapes (geopandas.GeoDataFrame): US state/DC polygons with a
            postal abbreviation column. Their CRS must match the raw
            coordinates because it is also assigned to respondent points.

    Returns:
        pandas.DataFrame: A copy with the same rows and index, renamed
        respondent columns, unordered categorical answers, padded string ZIP
        codes, and a boolean location_conflict column. Zero, missing, and
        unrecognized answer codes and invalid states become missing. No rows
        are dropped or imputed; coordinate values are retained unchanged.

    Raises:
        KeyError: Required columns or answer tables are absent.

    Notes:
        A location conflict requires distance above MAX_STATE_DISTANCE_KM
        (100 km) and a nearer state polygon. Distances use EPSG:5070 and
        coarse boundaries, so the flag is approximate, especially outside
        the contiguous US. Only nonmissing coordinate pairs are checked;
        coordinate ranges are not validated. False includes locations that
        could not be checked and does not certify correctness. ZIP conversion
        uses astype(str), so missing ZIP values would become strings.
    """
    cleaned_data = data.copy()

    for column in get_question_columns(cleaned_data):
        question_number = int(column[1:])
        answers = question_data["ans." + str(question_number)]
        labels = {}
        for position, answer in enumerate(answers["ans"]):
            labels[position + 1] = answer.strip()

        cleaned_data[column] = cleaned_data[column].map(labels)
        cleaned_data[column] = pd.Categorical(cleaned_data[column], categories=list(labels.values()), ordered=False)

    cleaned_data["ZIP"] = cleaned_data["ZIP"].astype(str).str.zfill(5)

    valid_states = list(state_shapes["postal"])
    invalid_state = ~cleaned_data["STATE"].isin(valid_states)
    cleaned_data.loc[invalid_state, "STATE"] = pd.NA

    has_location = cleaned_data[["lat", "long"]].notna().all(axis=1)
    located_data = cleaned_data.loc[has_location]
    points = gpd.points_from_xy(located_data["long"], located_data["lat"])
    points = gpd.GeoSeries(points, index=located_data.index, crs=state_shapes.crs)
    # Convert degrees to meters before measuring distances to state boundaries.
    points = points.to_crs(DISTANCE_CRS)
    projected_states = state_shapes.to_crs(DISTANCE_CRS)

    distances_km = pd.DataFrame(index=located_data.index)
    for row_number, state_row in projected_states.iterrows():
        distances_km[state_row["postal"]] = points.distance(state_row["geometry"]) / 1000
    closest_distance_km = distances_km.min(axis=1)

    cleaned_data["location_conflict"] = False
    for state in valid_states:
        in_state = located_data.index[located_data["STATE"] == state]
        state_distance_km = distances_km.loc[in_state, state]
        far_from_state = state_distance_km > MAX_STATE_DISTANCE_KM
        closer_to_another_state = state_distance_km > closest_distance_km.loc[in_state]
        cleaned_data.loc[in_state, "location_conflict"] = far_from_state & closer_to_another_state

    return cleaned_data.rename(columns=COLUMN_RENAMES)

def preprocess_data(cleaned_data, questions=None, require_location=False,
                    drop_location_conflicts=False, one_hot=False):
    """Select an analysis cohort and optionally one-hot encode its answers.

    Args:
        cleaned_data (pandas.DataFrame): Output of clean_data.
        questions (sequence[str] or None): Question columns to retain, in
            order. None selects all questions. Rows missing any selected
            answer are excluded, so the default is a complete-case cohort.
        require_location (bool): If True, also require both coordinates.
        drop_location_conflicts (bool): If True, exclude flagged conflicts.
            This does not exclude missing states or coordinates by itself.
        one_hot (bool): If True, return integer 0/1 columns named
            QNNN__answer. Retain every category, including unused choices.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: Answers and respondent
        metadata with identical retained row indices. Metadata includes the
        location flag. Inputs are not modified. One-hot encoding does not
        center, scale, impute, or include geographic variables.

    Raises:
        KeyError: Selected questions or required metadata columns are absent.

    Notes:
        Complete-case filtering can change the population represented in an
        analysis. Record exclusions and compare plausible alternatives before
        relying on the all-question cohort for dimension reduction/clustering.
    """
    if questions is None:
        questions = get_question_columns(cleaned_data)

    keep = cleaned_data[questions].notna().all(axis=1)

    if require_location:
        keep &= cleaned_data[["latitude", "longitude"]].notna().all(axis=1)

    if drop_location_conflicts:
        keep &= ~cleaned_data["location_conflict"]

    preprocessed_data = cleaned_data.loc[keep]
    answers = preprocessed_data[questions].copy()
    respondents = preprocessed_data[RESPONDENT_COLUMNS].copy()

    if one_hot:
        answers = pd.get_dummies(answers, prefix_sep="__", dtype=int)

    return answers, respondents
