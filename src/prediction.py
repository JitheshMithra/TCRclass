import torch
import esm
import numpy as np
import pandas as pd
from model import TCRClassifier
import os
import pickle


testdataframe=pd.read_csv("../data/test_set.csv")
print(f"Test set shape: {testdataframe.shape}")
print(f"Test columns: {testdataframe.columns.to_list()}")

with open('../outputs/vencoder.pkl','rb') as f:
    vencoder= pickle.load(f)
with open('../outputs/jencoder.pkl','rb') as f:
    jencoder =pickle.load(f)

print("\nLoading ESM-2...")
modelesm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batchconverter=alphabet.get_batch_converter()
modelesm.eval()

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
    
modelesm=modelesm.to(device)

def getembedding(sequence):
    data=[("seq", sequence)]
    batchlabels, batchstrs, batchtokens=batchconverter(data)
    batchtokens=batchtokens.to(device)
    with torch.no_grad():
        results=modelesm(batchtokens, repr_layers=[33], return_contacts=False)
    tokenembedding=results["representations"][33]
    embedding=tokenembedding[0,1:len(sequence)+1].mean(0)
    return embedding.cpu().numpy()
print("\nGenerating test embeddings...")
testembeddings=[]
batchsize=32

for i in range(0, len(testdataframe), batchsize):
    batch=testdataframe.iloc[i:i+batchsize]
    for k in batch["CDR3.beta.aa"]:
        emboy=getembedding(k)
        testembeddings.append(emboy)
    if (i//batchsize)%10==0:
        print(f"Processed {min(i+batchsize,len(testdataframe))}/{len(testdataframe)} sequences")
testembeddings=np.array(testembeddings)
print(f"\nTest embeddings shape: {testembeddings.shape}")

testdataframe['TRBV']=testdataframe['TRBV'].fillna('unknown')
testdataframe['TRBJ']= testdataframe['TRBJ'].fillna('unknown')

#handle unseen gene values
def safetransform(encoder, values):
    known = set(encoder.classes_)
    return np.array([encoder.transform([v])[0] if v in known else encoder.transform(['unknown'])[0] for v in values])

vtest = safetransform(vencoder,testdataframe['TRBV']).reshape(-1, 1)
jtest = safetransform(jencoder,testdataframe['TRBJ']).reshape(-1, 1)

testembeddings = np.concatenate([testembeddings,vtest,jtest], axis=1)

print(f"Test embeddings with V/J shape: {testembeddings.shape}")

classifier=TCRClassifier().to(device)
classifier.load_state_dict(torch.load("../outputs/model.pt", map_location=device))
classifier.eval()

print("\nRunning predictions...")
xtensor=torch.tensor(testembeddings,dtype=torch.float32).unsqueeze(1).to(device)

with torch.no_grad():
    output=classifier(xtensor)
    #lowk this variable name is hella sus
    preds =torch.argmax(output,dim=1).cpu().numpy()
labelmap={0:'autoimmune',1:'bacterial',2:'cancer',3:'viral'}
predlabels = []
for p in preds:
    predlabels.append(labelmap[p])
    
print((f"\nPrediction distribution:"))
print(pd.Series(predlabels).value_counts())

os.makedirs("../outputs", exist_ok=True)
submission=pd.DataFrame({
    "ID": testdataframe["ID"],
    "prediction": predlabels
})
submission.to_csv("../outputs/submission.csv", index=False)
print("\nSubmission saved to ../outputs/submission.csv")

 