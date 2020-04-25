import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
class PersonEmbeding(nn.Module):
    def __init__(self, nb_emb=4, back_bone='resnet34'):
        super(PersonEmbeding, self).__init__()
        pt_model = torch.hub.load('pytorch/vision:v0.5.0', back_bone, pretrained=True)
        self.encoder = nn.Sequential(*list(pt_model.children())[:-1])
        self.encoder.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.preprocess_training = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.nb_emb = nb_emb
    def forward(self, inputs):
        with torch.no_grad():
            emb = self.encoder(inputs)
            emb = emb.view(emb.size(0), -1)
            return emb

    def gen_training_emb(self, images):
        batch = []
        for img in images:
            img = Image.fromarray(img)
            for _ in range(self.nb_emb):
                batch.append(self.preprocess_training(img))
        return self.forward(torch.stack(batch))
    
    def embed(self, images):
        batch = [self.preprocess(Image.fromarray(img)) for img in images]
        return self.forward(torch.stack(batch))
            

