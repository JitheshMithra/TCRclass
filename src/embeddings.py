import torch
import esm
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle

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

#v and j genes
dataframe["V_gene"]=dataframe["V_gene"].fillna("unknown")
dataframe["J_gene"]=dataframe["J_gene"].fillna("unknown")

vencoder=LabelEncoder()
jencoder=LabelEncoder()

vencoded=vencoder.fit_transform(dataframe["V_gene"]).reshape(-1,1)
jencoded=jencoder.fit_transform(dataframe["J_gene"]).reshape(-1,1)

embeddings=np.concatenate([embeddings, vencoded, jencoded], axis=1)
print(f"\nEmbeddings with V/J shape: {embeddings.shape}")

os.makedirs("../outputs", exist_ok=True)
with open("../outputs/vencoder.pkl","wb") as f:
    pickle.dump(vencoder,f)
with open("../outputs/jencoder.pkl","wb") as f:
    pickle.dump(jencoder,f)

np.save("../outputs/embeddings/esm2_embeddings.npy",embeddings)
dataframe.to_csv("../outputs/embeddings/cleaned_with_labels.csv",index=False)
print("Saved embeddings with V/J to ../outputs/embeddings/esm2_embeddings.npy")