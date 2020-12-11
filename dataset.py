import h5py
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, file):
        super(TrainDataset, self).__init__()
        self.file = file

    def __getitem__(self, index):
        with h5py.File(self.file, 'r') as f:
            return (f['lr'][index] / 255. )[np.newaxis, :], (f['hr'][index] / 255. )[np.newaxis, :]
            # return np.expand_dims(f['lr'][index] / 255., 0), np.expand_dims(f['hr'][index] / 255., 0)

    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            return len(f['lr'])

# class EvalDataset(TrainDataset):
#     def __getitem__(self, idx):
#         super(EvalDataset, self).__getitem__()
#         with h5py.File(self.h5_file, 'r') as f:
#             return (f['lr'][str(idx)][:, :] / 255.)[np.newaxis, :], (f['hr'][str(idx)][:, :] / 255.)[np.newaxis, :]

class EvalDataset(Dataset):
    def __init__(self, file):
        super(EvalDataset, self).__init__()
        self.file = file

    def __getitem__(self, index):
        with h5py.File(self.file, 'r') as f:
            return (f['lr'][str(index)][:, :] / 255.)[np.newaxis, :], (f['hr'][str(index)][:, :] / 255.)[np.newaxis, :]

    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            return len(f['lr'])