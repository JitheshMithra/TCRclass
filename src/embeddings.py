import torch
import esm
import pandas as pd
import numpy as np
import os

dataframe=pd.read_csv("../data/TCR-cleaned.csv")
print(f"loaded {len(dataframe)} sequences")

print("\nloading ESM model...")
model, alphabet=esm.pretrained.esm2_t33_650M_UR50D()
batchconverter=alphabet.get_batch_converter()
model.eval()

#move to gpu 
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
model = model.to(device)

def getembedding(sequence):
    data=[("seq", sequence)]
    batchlabels, batchstrs, batchtokens = batchconverter(data)
    batchtokens = batchtokens.to(device)
    with torch.no_grad():
        results = model(batchtokens, repr_layers=[33], return_contacts=False)
  
    tokenembeddings = results["representations"][33]
    #no bos or eos 
    embedding= tokenembeddings[0,1:len(sequence)+1].mean(0)
    return embedding.cpu().numpy()

print("Generating embeddings...")
embeddings=[]
batchsize=32

for i in range(0, len(dataframe), batchsize):
    batch=dataframe.iloc[i:i+batchsize]
    batchembeddings=[]
    for k in batch['CDR3']:
        emb=getembedding(k)
        batchembeddings.append(emb)
    embeddings.extend(batchembeddings)
    if (i//batchsize)%10==0:
        print(f"processed {min(i+batchsize, len(dataframe))} / {len(dataframe)} sequences")

embeddings=np.array(embeddings)
print(f"\nEmbeddings shape: {embeddings.shape}")

os.makedirs("../outputs/embeddings", exist_ok=True)
np.save("../outputs/embeddings/esm2_embeddings.npy", embeddings)
dataframe.to_csv("../outputs/embeddings/cleaned_with_labels.csv", index=False)
print("Saved embeddings to ../outputs/embeddings/esm2_embeddings.npy")
print("Saved labels to ../outputs/embeddings/cleaned_with_labels.csv")