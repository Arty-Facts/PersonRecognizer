import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import tqdm
from pathlib import Path
from random import choice

class EMB_Dataset(Dataset):
    def __init__(self, emb_maniger, name):
        self.db = emb_maniger
        self.others = [str(dset) for dset in emb_maniger.info if str(dset) != name]
        self.name = name
        self.size = emb_maniger.get_len(name)
    def __len__(self):
        return self.size*2

    def __getitem__(self, idx):
        if idx < self.size:
            return self.db.get(self.name, idx), torch.tensor(1, dtype=torch.long)
        return self.db.get_random(choice(self.others)), torch.tensor(0, dtype=torch.long)
class Recognizer(nn.Module):
    def __init__(self,name, model_dir="models", load=True,  emb=512, out=2, batch_size=16, lr=1e-4):
        super(Recognizer, self).__init__()
        _dir = Path(model_dir)
        if not _dir.is_dir():
            _dir.mkdir()
        self.name = name
        self.model_dir = model_dir
        self.model = nn.Sequential(
                        nn.Linear(emb, out),
                        nn.Softmax(dim=1),
                    )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.bs = batch_size
        if load:
            self.load()
    
    def save(self):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, f"{self.model_dir}/{self.name}.ckpt")
        print(f"Model saved at: {self.model_dir}/{self.name}.ckpt")

    def load(self):
        if Path(f"{self.model_dir}/{self.name}.ckpt").exists():
            print(f"Loading {self.model_dir}/{self.name}.ckpt")
            ckpt = torch.load(f"{self.model_dir}/{self.name}.ckpt")
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        elif Path(f"{self.model_dir}/Otrher.ckpt").exists():
            print(f"{self.model_dir}/{self.name}.ckpt not found ...")
            print(f"Starting with pretrined models weights")
            ckpt = torch.load(f"{self.model_dir}/Others.ckpt")
            self.model.load_state_dict(ckpt["model"])
        else:
            print(f"{self.model_dir}/{self.name}.ckpt not found ...")
            print(f"Starting with random weights")

    def forward(self, inputs):
        with torch.no_grad():
            return self.model(inputs)

    def get_beter(self, emb_maniger):
        if self.name not in emb_maniger.info:
            print(f"No data for {self.name} in Embeding Maniger")
            return 
        loss_func = nn.CrossEntropyLoss()
        ref_loss = float("inf")
        data = EMB_Dataset(emb_maniger, self.name)
        tot = len(data)
        data = DataLoader(data, shuffle=True, batch_size=self.bs)
        while True:
            tot_loss = 0
            for x,y in data:
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = loss_func(pred, y)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()/self.bs
            curr_loss = tot_loss/tot
            if ref_loss < curr_loss:
                break
            ref_loss = curr_loss
        print(f"curret loss:{curr_loss}")




