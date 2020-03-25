def remove_rows_in_col(df, column_name: str, accepted_chars: list, max_length: int = 40):
    """
    all rows removed from a column that do not
    contain accepted characters
    data_frame: Panda dataframe to be filtered
    column_name: The column name in df to be filtered through
    accepted_chars: List of characters that are acceptable in df row
    """
    # WARNING: The line below used to be len(x) <= max_length
    return df[df[column_name].apply(
        lambda x: set(x).issubset(accepted_chars) and len(x) == max_length if (isinstance(x, str)) else True)]


def convert_row_to_lower(data_frame, column_name: str):
    """
    Converts all characters in data_frame column to lowercase
    data_frame: Panda dataframe to be filtered
    column_name: The colmun name of the df to be filtered through
    """
    data_frame[column_name] = data_frame[column_name].map(lambda x: x.lower())
