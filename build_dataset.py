import random
import segyio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SegyDataset(Dataset):
    def __init__(self, segy_file, mode, img_size):
        segy_handle = segyio.open(segy_file, ignore_geometry=True, strict=False)
        self.alltraces = segy_handle.trace.raw[:]
        iltraces = segy_handle.attributes(segyio.TraceField.INLINE_3D)[:]
        xltraces = segy_handle.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        self.tracedf = pd.DataFrame(data={'IL':iltraces, 'XL':xltraces})
        uniqueIL = np.unique(iltraces)
        uniqueXL = np.unique(xltraces)
        self.mode = mode
        self.seistraces = self.alltraces.shape[0]
        self.seisdepth = self.alltraces.shape[1]
        self.imgsize = img_size
        if self.mode == "train":
            self.num_images = 1024
            self.idx_nums = uniqueIL[-self.num_images//2-10:-10]
        else:
            self.num_images = 128
            self.idx_nums = uniqueIL[:self.num_images//2]

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if torch.rand(1) > 0.5:
            # Return random traces
            nonseis = random.sample(range(self.seistraces), self.imgsize)
            randline = self.alltraces[nonseis, :]
            startdep = random.sample(range(self.seisdepth-self.imgsize), 1)[0]
            image = randline.T[startdep:startdep+self.imgsize,:].astype(np.float32)
            np.random.shuffle(image)
            label = torch.tensor(0)
        else:
            # Return seismic
            ilnum = self.idx_nums[idx//2]
            trnums = self.tracedf[self.tracedf['IL']==ilnum].index.values
            seisline = self.alltraces[trnums, :]
            startdep = random.sample(range(self.seisdepth-self.imgsize), 1)[0]
            startx = random.sample(range(len(trnums)-self.imgsize), 1)[0]
            image = seisline.T[startdep:startdep+self.imgsize, startx:startx+self.imgsize].astype(np.float32)
            label = torch.tensor(1)
        image = (image - np.mean(image)) / np.std(image)
        image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image), label
