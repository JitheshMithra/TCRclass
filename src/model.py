import torch
import torch.nn as nn

class TCRClassifier(nn.Module):
    def __init__(self, embeddingdim=1280, numheads=8, numclasses=4,dropout=0.1):
        super(TCRClassifier, self).__init__()
        self.attention=nn.MultiheadAttention(embed_dim=embeddingdim, num_heads=numheads,dropout=dropout,batch_first=True)
        self.layernorm=nn.LayerNorm(embeddingdim)
        self.classifier=nn.Sequential(nn.Linear(embeddingdim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, numclasses))
    
    def forward(self, x):
        attnoutput, _=self.attention(x,x,x)
        x=self.layernorm(x+attnoutput)
        x=x.squeeze(1)
        output=self.classifier(x)
        return output