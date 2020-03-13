import torch.nn as nn
class Recognizer(nn.Module):

    def __init__(self, emb=512, out=2):
        super(Recognizer, self).__init__()
        self.layer = nn.Sequential(
                        nn.Linear(emb, out),
                        nn.Softmax(dim=1),
                    )

    def forward(self, inputs):
        return self.layer(inputs)