from torch.utils.data import Dataset
import unicodedata
import math

class NameDataset(Dataset):
    def __init__(self, df):
        self.data_frame = df

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        name = self.data_frame['name'].iloc[index]
        first = self.data_frame['first'].iloc[index]
        middle = self.data_frame['middle'].iloc[index]
        last = self.data_frame['last'].iloc[index]

        if isinstance(first, float):
            first = ''

        if isinstance(middle, float):
            middle = ''

        if isinstance(last, float):
            last = ''


        return name, first, middle, last
