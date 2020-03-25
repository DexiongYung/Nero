from torch.utils.data import Dataset

from .util_helper import remove_rows_in_col, convert_row_to_lower


class NameDataset(Dataset):
    def __init__(self, df, col_name, character_set, max_name_length=40):
        """
        Args:
            csv_file (string): Path to the csv file WITHOUT labels
            col_name (string): The column name corresponding to the people names that'll be standardized
        """
        self.character_set = character_set
        if 0 < len(df):
            df = self._clean_dataframe(df, col_name, max_name_length)
        self.data_frame = df[col_name]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return self.data_frame.iloc[index]

    def add_df(self, df, col_name):
        """
        Args:
            df (string): Path to dataframe file WITHOUT labels to be added to self.data_frame
            col_name (string): Name of column in csv with name
        """
        df = self._clean_dataframe(df, col_name)
        self.data_frame = self.data_frame.append(df[col_name]).drop_duplicates()

    def _clean_dataframe(self, df, col_name, max_name_length):
        if len(df) > 0:
            convert_row_to_lower(df, col_name)
            df = remove_rows_in_col(df, col_name, self.character_set, max_length=max_name_length)
        return df
