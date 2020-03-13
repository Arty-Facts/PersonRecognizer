import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
import tqdm

class EMB_Dataset(Dataset):
    def __init__(self, true_data, false_data):
        self.true_data = true_data
        self.false_data = false_data
        self.offset = len(self.true_data)-1
    def __len__(self):
        return len(self.true_data) + len(self.false_data) - 1

    def __getitem__(self, idx):
        if idx < len(self.true_data):
            return self.true_data[idx], torch.tensor(1, dtype=torch.long)
        return self.false_data[idx-self.offset], torch.tensor(0, dtype=torch.long)
class Recognizer(nn.Module):

    def __init__(self, emb=512, out=2, batch_size=16, lr=1e-4):
        super(Recognizer, self).__init__()
        self.layer = nn.Sequential(
                        nn.Linear(emb, out),
                        nn.Softmax(dim=1),
                    )
        self.optimizer = optim.Adam(self.layer.parameters(), lr=lr)
        self.bs = batch_size

    def forward(self, inputs):
        with torch.no_grad():
            return self.layer(inputs)

    def get_beter(self, true_data, false_data):
        loss_func = nn.CrossEntropyLoss()
        data = EMB_Dataset(true_data,false_data)
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




