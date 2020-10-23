import torch.utils.data as data
import h5py
import torch


class SlideLevelDataset(data.Dataset):
    def __init__(self, hdf5_path, class_info):
        super(SlideLevelDataset, self).__init__()
        self.class_info = class_info
        self.hdf5_path = hdf5_path

        with h5py.File(hdf5_path, "r") as hdf5_file:
            self.keys = list(hdf5_file.keys())

        sample_ids = [key[:15] for key in self.keys]
        assert sample_ids == list(self.class_info.keys())

    def __getitem__(self, index):
        hdf5_file = h5py.File(self.hdf5_path, "r")
        key = self.keys[index]
        target = self.class_info[key[:15]]
        features = hdf5_file['{:s}/features'.format(key)][()]
        indices = hdf5_file['{:s}/indices'.format(key)][()]

        return torch.tensor(features).float(), torch.tensor([target]).long(), indices

    def __len__(self):
        return len(self.keys)




