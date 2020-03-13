from pathlib import Path
import h5py
from matplotlib import pyplot as plt
import numpy as np
from random import randint

class EmbedingsManiger():
    def __init__(self, db_dir="db", db="embedings"):
        self.db_dir = Path("db")
        if not self.db_dir.is_dir():
            self.db_dir.mkdir()
        self.db = f"{str(self.db_dir)}/{db}.hdf5"
        with h5py.File(self.db, "a", libver='latest', swmr=True) as f:
            self.info = set([ str(item) for item in f.keys()])

    def get_len(self, name):
        if name not in self.info:
            print("Not in db...")
            return 0
        with h5py.File(self.db, "r", libver='latest', swmr=True) as f:
            return f[name]["emb"].shape[0]

    def get(self, name, idx):
        with h5py.File(self.db, "r", libver='latest', swmr=True) as f:
            return np.asarray(f[name]["emb"][idx])

    def get_random(self, name):
        with h5py.File(self.db, "a", libver='latest', swmr=True) as f:
            idx = randint(0,len(f[name]["emb"])-1)
            return np.asarray(f[name]["emb"][idx, :])

    def new(self, name, image, data):
        if name in self.info:
            return False
        else:
            data = np.array([d.numpy() for d in data])
            with h5py.File(self.db, "a", libver='latest', swmr=True) as f:
                root = f.create_group(name)
                root.create_dataset("image", data=image)#, compression="lzf")
                root.create_dataset("emb", data=data, maxshape=(None,512))#, compression="lzf")
            self.info.add(name)
            return True

    def remove(self,name):
        with h5py.File(name, "a", libver='latest', swmr=True) as f:
            del f[name]
            self.info.remove(name)
        return True

    def show(self, name):
        if name in self.info:
            with h5py.File(self.db, "r", libver='latest', swmr=True) as f:
                image = np.asarray(f[name]["image"])
            plt.imshow(image)
            plt.show
        else:
            print("Not in db...")

    def add(self, name, data):
        if name not in self.info:
            print("Not in db...")
            return False
        data = data.numpy()
        if len(data.shape) == 1:
            data = data[np.newaxis, ... ]
        with h5py.File(self.db, "a", libver='latest', swmr=True) as f:
            dset = f[name]["emb"]
            dset.resize(dset.shape[0]+data.shape[0], axis=0) 
            dset[-data.shape[0]:] = data
        return True