from torch.utils.data import Dataset
import unicodedata

class NameDataset(Dataset):
    def __init__(self, df):
        self.data_frame = df

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        name = unicodedata.normalize('NFKD', self.data_frame['name']).encode('ascii','ignore')
        first = unicodedata.normalize('NFKD', self.data_frame['first']).encode('ascii','ignore')
        middle = unicodedata.normalize('NFKD', self.data_frame['middle']).encode('ascii','ignore')
        last = unicodedata.normalize('NFKD', self.data_frame['last']).encode('ascii','ignore')

        return name, first, middle, last
