import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
class PersonEmbeding(nn.Module):
    def __init__(self):
        super(PersonEmbeding, self).__init__()
        pt_model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained=True)
        self.encoder = nn.Sequential(*list(pt_model.children())[:-1])
        self.encoder.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, inputs):
        emb = self.encoder(inputs)
        emb = emb.view(emb.size(0), -1)
        return emb
    
    def embed(self, images):
        batch = [self.preprocess(Image.fromarray(img)) for img in images]
        return self.forward(torch.stack(batch))
            

