import torch
import esm
import numpy as np
import matplotlib.pyplot as plt
import os
from model import TCRClassifier

sequences = {'viral': 'CASSLAPGATNEKLFF','bacterial': 'CASSLGQAYEQYF','cancer': 'CASSIRSSYEQYF','autoimmune': 'CASSPGTSGNTIYF'}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Loading ESM-2...")
modelesm, alphabet=esm.pretrained.esm2_t33_650M_UR50D()
batchconverter=alphabet.get_batch_converter()
modelesm.eval()
modelesm=modelesm.to(device)

classifier=TCRClassifier().to(device)
classifier.load_state_dict(torch.load("../outputs/model.pt", map_location=device))
classifier.eval()

def getembedding(sequence):
    data=[("seq", sequence)]
    batchlabels, batchstrs, batchtokens = batchconverter(data)
    batchtokens = batchtokens.to(device)
    with torch.no_grad():
        results = modelesm(batchtokens, repr_layers=[33], return_contacts=False)
    tokenembeddings = results["representations"][33]
    #return pertoken embeddings, not mean pooled
    return tokenembeddings[0, 1:len(sequence)+1].cpu().numpy()

def getattention(tokenembeddings):
    global avgattn
    x = torch.tensor(tokenembeddings, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        attnoutput, attnweights = classifier.attention(x, x, x, average_attn_weights=False)
    avgattn = attnweights.squeeze(0).mean(axis=0)
    return avgattn.cpu().numpy()
os.makedirs("../outputs/figures", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes=axes.flatten()

for idx,(classname, seq) in enumerate(sequences.items()):
    tokenembs =getembedding(seq)
    attnweights= getattention(tokenembs)
    
    ax = axes[idx]
    im = ax.imshow(attnweights, cmap='Blues')
    ax.set_xticks(range(len(seq)))
    ax.set_yticks(range(len(seq)))
    ax.set_xticklabels(list(seq), fontsize=8)
    ax.set_yticklabels(list(seq), fontsize=8)
    ax.set_title(f"{classname}\n{seq}", fontsize=10)
    plt.colorbar(im, ax=ax)

plt.suptitle("Attention Heatmaps by Disease Class", fontsize=14)
plt.tight_layout()
plt.savefig("../outputs/figures/attention_heatmaps.png", dpi=150)
print("Saved attention heatmaps to ../outputs/figures/attention_heatmaps.png")