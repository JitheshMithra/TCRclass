import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from model import TCRClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc

embeddings=np.load("../outputs/embeddings/esm2_embeddings.npy")
dataframe=pd.read_csv("../outputs/embeddings/cleaned_with_labels.csv")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Label distribution:\n{dataframe['label'].value_counts()}")

#encode the labels to the integers
labelencoder=LabelEncoder()
labels=labelencoder.fit_transform(dataframe["label"])
print(f"\nLabel mapping: {dict(zip(labelencoder.classes_, range(len(labelencoder.classes_))))}")

#train test split
xtrain, xval, ytrain, yval = train_test_split(embeddings,labels,test_size=0.2,random_state=42,stratify=labels)
print(f"\nTrain size: {len(xtrain)}, Val size: {len(xval)}")

#class weights for weighted loss (pretty important yo)
classcount=np.bincount(ytrain)
classweights=1.0/classcount
classweights=classweights/classweights.sum()*len(classcount)
classweights=torch.tensor(classweights,dtype=torch.float32)
print(f"\nClass weights: {classweights}")

class TCRDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

trainset=TCRDataset(xtrain,ytrain)
valset=TCRDataset(xval,yval)
trainloader=DataLoader(trainset,batch_size=64,shuffle=True)
valloader=DataLoader(valset,batch_size=64,shuffle=False)

#main setup
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

model=TCRClassifier().to(device)
classweights=classweights.to(device)
criterion=nn.CrossEntropyLoss(weight=classweights)
optimizer=torch.optim.Adam(model.parameters(),lr=5e-5,weight_decay=1e-4)
bestval = float('inf')
patience = 10
patientcounter = 0
scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,factor=0.5)
epochs= 50
trainlosses=[]
vallosses=[]
valf1s=[]
for i in range(epochs):
    model.train()
    runningloss=0.0
    
    for xbatch,ybatch in trainloader:
        xbatch=xbatch.unsqueeze(1).to(device)
        ybatch =ybatch.to(device)
        optimizer.zero_grad()
        output =model(xbatch)
        loss=criterion(output, ybatch)
        loss.backward()
        optimizer.step()
        runningloss += loss.item()
    trainloss=runningloss/len(trainloader)
    trainlosses.append(trainloss)
    
    #valdation
    model.eval()
    runningvalloss=0.0
    allpreds=[]
    alltrue=[]
    with torch.no_grad():
        for xbatch,ybatch in valloader:
            xbatch=xbatch.unsqueeze(1).to(device)
            ybatch =ybatch.to(device)
            output =model(xbatch)
            loss=criterion(output, ybatch)
            runningvalloss += loss.item()
            preds=torch.argmax(output,dim=1).cpu().numpy()
            allpreds.extend(preds)
            alltrue.extend(ybatch.cpu().numpy())
    valloss=runningvalloss/len(valloader)
    vallosses.append(valloss)
    scheduler.step(valloss)
    if valloss < bestval:
        bestval= valloss
        torch.save(model.state_dict(), "../outputs/model.pt")
        patientcounter = 0
    else:
        patientcounter += 1
        if patientcounter>= patience:
            print(f"Early stopping at epoch {i+1}")
            break
    f1=f1_score(alltrue, allpreds, average="macro")
    valf1s.append(f1)
    print(f"Epoch {i+1}/{epochs} | Train Loss: {trainloss:.4f} | Val Loss: {valloss:.4f} | Macro F1: {f1:.4f}")
print("\nModel saved to ../outputs/model.pt")
os.makedirs("../outputs/figures", exist_ok=True)

#load model for final evaluation
model.load_state_dict(torch.load("../outputs/model.pt"))
model.eval()

finalpreds = []
finaltrue = []
finalprobs = []

with torch.no_grad():
    for xbatch, ybatch in valloader:
        xbatch = xbatch.unsqueeze(1).to(device)
        ybatch = ybatch.to(device)
        output = model(xbatch)
        probs = torch.softmax(output, dim=1).cpu().numpy()
        preds = torch.argmax(output, dim=1).cpu().numpy()
        finalpreds.extend(preds)
        finaltrue.extend(ybatch.cpu().numpy())
        finalprobs.extend(probs)

finalpreds = np.array(finalpreds)
finaltrue = np.array(finaltrue)
finalprobs = np.array(finalprobs)

#per class f1
print("\nPer-class F1:")
print(classification_report(finaltrue, finalpreds, target_names=labelencoder.classes_))

#confusion matrix
plt.figure(figsize=(8,6))
cm =confusion_matrix(finaltrue, finalpreds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labelencoder.classes_, yticklabels=labelencoder.classes_, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("../outputs/figures/confusion_matrix.png")
print("Saved confusion matrix")

#AUC curves
finaltruebinary = label_binarize(finaltrue, classes=[0,1,2,3])
plt.figure(figsize=(8,6))
colors = ['blue', 'red', 'green', 'orange']
for i, (classname, color) in enumerate(zip(labelencoder.classes_, colors)):
    fpr, tpr, _ = roc_curve(finaltruebinary[:,i], finalprobs[:,i])
    aucval = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label=f"{classname} (AUC={aucval:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/roc_curves.png")
print("Saved ROC curves")

#loss curves just for you yash
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(trainlosses, label="Train Loss")
plt.plot(vallosses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.subplot(1,2,2)
plt.plot(valf1s, label="Val Macro F1", color="green")
plt.xlabel("Epoch")
plt.ylabel("Macro F1 Score")
plt.title("Validation Macro F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/training_curves.png")       
print("\nTraining curves saved to ../outputs/figures/training_curves.png")