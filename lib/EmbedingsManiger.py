from pathlib import Path
import h5py
from matplotlib import pyplot as plt
import numpy as np
from random import randint

class EmbedingsManiger():
    def __init__(self, db_dir="db", db="embedings", cache_ram=False):
        self.db_dir = Path("db")
        if not self.db_dir.is_dir():
            self.db_dir.mkdir()
        self.db = f"{str(self.db_dir)}/{db}.hdf5"
        self.cache_ram = cache_ram
        with h5py.File(self.db, "a", libver='latest', swmr=True) as f:
            self.info = set([ str(item) for item in f.keys()])
            if cache_ram:
                self.cache_db()
    def cache_db(self):
        with h5py.File(self.db, "a", libver='latest', swmr=True) as f:
            self.cashed_db = {name: 
                                {
                                    "image": np.asarray(f[name]["image"]),
                                    "emb": np.asarray(f[name]["emb"])
                                } for name in self.info}

    def get_len(self, name):
        if name not in self.info:
            print("Not in db...")
            return 0
        if self.cache_ram:
            return self.cashed_db[name]["emb"].shape[0]

        with h5py.File(self.db, "r", libver='latest', swmr=True) as f:
            return f[name]["emb"].shape[0]

    def get(self, name, idx):
        if self.cache_ram:
            return np.asarray(self.cashed_db[name]["emb"][idx])

        with h5py.File(self.db, "r", libver='latest', swmr=True) as f:
            return np.asarray(f[name]["emb"][idx])

    def get_random(self, name):
        if self.cache_ram:
            idx = randint(0,len(self.cashed_db[name]["emb"])-1)
            return np.asarray(self.cashed_db[name]["emb"][idx, :])

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
            if self.cache_ram:
                self.cache_db()
            return True

    def remove(self,name):
        with h5py.File(self.db, "a", libver='latest') as f:
            del f[name]
            self.info.remove(name)
        if self.cache_ram:
            self.cache_db()
        return True

    def show(self, name):
        if name in self.info:
            if self.cache_ram:
                image = np.asarray(self.cashed_db[name]["image"])
            else:
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
        data = np.array([d.numpy() for d in data])
        with h5py.File(self.db, "a", libver='latest', swmr=True) as f:
            dset = f[name]["emb"]
            dset.resize(dset.shape[0]+data.shape[0], axis=0) 
            dset[-data.shape[0]:] = data
        if self.cache_ram:
            self.cache_db()
        return True