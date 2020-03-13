import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import tqdm
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

    def __init__(self,name, emb=512, out=2, batch_size=16, lr=1e-4):
        super(Recognizer, self).__init__()
        self.name = name
        self.layer = nn.Sequential(
                        nn.Linear(emb, out),
                        nn.Softmax(dim=1),
                    )
        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr)
        self.bs = batch_size

    def forward(self, inputs):
        with torch.no_grad():
            return self.layer(inputs)

    def get_beter(self, emb_maniger):
        if self.name not in emb_maniger.info:
            print(f"No data for {self.name} in Embeding Maniger")
            return 
        loss_func = nn.CrossEntropyLoss()
        data = EMB_Dataset(emb_maniger, self.name)
        tot = len(data)
        data = DataLoader(data, shuffle=True, batch_size=self.bs)
        tot_loss = 0
        for x,y in data:
            self.optimizer.zero_grad()

            pred = self.layer(x)
            loss = loss_func(pred, y)
            loss.backward()
            self.optimizer.step()
            tot_loss += loss.item()/self.bs
        print(f"curret loss:{tot_loss/tot}")




