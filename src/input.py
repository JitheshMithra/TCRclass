import torch
import esm
import numpy as np
import pickle
import argparse
from model import TCRClassifier

parser =argparse.ArgumentParser()
parser.add_argument('--cdr3',type=str,required=True)
parser.add_argument('--vgene', type=str, default='unknown')
parser.add_argument('--jgene', type=str, default='unknown')
args=parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"\nLoading ESM-2...")
modelesm, alphabet= esm.pretrained.esm2_t33_650M_UR50D()
batchconverter =alphabet.get_batch_converter()
modelesm.eval()
modelesm= modelesm.to(device)

def getembedding(sequence):
    data=[("seq", sequence)]
    batchlabels, batchstrs, batchtokens = batchconverter(data)
    batchtokens=batchtokens.to(device)
    with torch.no_grad():
        results=modelesm(batchtokens, repr_layers=[33], return_contacts=False)
    tokenembeddings= results["representations"][33]
    embedding= tokenembeddings[0, 1:len(sequence)+1].mean(0)
    return embedding.cpu().numpy()

with open('../outputs/vencoder.pkl', 'rb') as f:
    vencoder =pickle.load(f)
with open('../outputs/jencoder.pkl', 'rb') as f:
    jencoder= pickle.load(f)

def safeencode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return encoder.transform(['unknown'])[0]

print(f"Processing sequence: {args.cdr3}")
embedding=getembedding(args.cdr3)
venc=np.array([safeencode(vencoder, args.vgene)])
jenc= np.array([safeencode(jencoder, args.jgene)])
combined=np.concatenate([embedding, venc, jenc])

classifier= TCRClassifier(embeddingdim=1282).to(device)
classifier.load_state_dict(torch.load('../outputs/model.pt', map_location=device))
classifier.eval()

xtensor=torch.tensor(combined, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad():
    output=classifier(xtensor)
    probs=torch.softmax(output, dim=1).cpu().numpy()[0]
    pred=np.argmax(probs)

labelmap={0:'autoimmune',1:'bacterial',2:'cancer', 3:'viral'}

print(f"\n{'='*40}")
print(f"CDR3:{args.cdr3}")
print(f"V gene:{args.vgene}")
print(f"J gene:{args.jgene}")
print(f"{'='*40}")
print(f"Prediction: {labelmap[pred].upper()}")
print(f"{'='*40}")
print(f"autoimmune:{probs[0]:.4f}")
print(f"bacterial:{probs[1]:.4f}")
print(f"cancer:{probs[2]:.4f}")
print(f"viral: {probs[3]:.4f}")
print(f"{'='*40}")