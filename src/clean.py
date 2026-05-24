import pandas as pd
import numpy as np

#load that data boyyyyy
dataframe = pd.read_csv("../data/TCR-Processed-Raw.csv")

print(f"Original Shape: {dataframe.shape}")
print(f"\nPathology Values: \n{dataframe['Pathology'].value_counts().to_string()}")

#the worst part: label mapping
label_map = {
    #viral
    'Influenza':'viral','Cytomegalovirus (CMV)':'viral','Epstein Barr virus (EBV)':'viral','Human immunodeficiency virus (HIV)':'viral','Yellow fever virus':'viral','HTLV-1': 'viral','Hepatitis C virus':'viral','COVID-19': 'viral','Herpes simplex virus 2 (HSV2)':'viral','Hepatitis E virus infection (cHEV)':'viral','ARDS':'viral',
    #bacterial
    'M. tuberculosis':'bacterial',
    #cancer
    'Melanoma':'cancer', 'Tumor associated antigen (TAA)':'cancer', 'Colorectal cancer':'cancer','Breast Cancer':'cancer','Lung cancer':'cancer','Leukemia':'cancer','Neoantigen':'cancer','Acute myeloid leukemia':'cancer','Carcinoma':'cancer','Epithelial ovarian cancer':'cancer',
    #autoimmune
    'Allergy':'autoimmune',"Alzheimer's disease":'autoimmune','Parkinson disease':'autoimmune','Multiple sclerosis (MS)':'autoimmune','Diabetes Type 1':'autoimmune','Psoriatic arthritis':'autoimmune','Churg Strauss syndrome':'autoimmune','Toxic epidermal necrolysis':'autoimmune','Biliary atresia':'autoimmune',
}

#pathology is null so we drop that
dataframe=dataframe.dropna(subset=['Pathology'])
print(f"\nShape after dropping null Pathology: {dataframe.shape}")

#same with CAS because its only 26 rows
dataframe=dataframe[dataframe['Pathology'] != 'Calcified Aortic Stenosis disease']

#apply label map
dataframe['label']=dataframe['Pathology'].map(label_map)

#check unmapped
unmapped=dataframe[dataframe['label'].isna()]['Pathology'].value_counts()
if len(unmapped)>0:
    print(f"\n Unmapped Pathologies (will be dropped): \n{unmapped}")
else:
    print("\nAll pathologies have been mapped successfully")
#rename colums
dataframe=dataframe.rename(columns={
    'CDR3.beta.aa':'CDR3','TRBV':'V_gene','TRBJ':'J_gene',
})
#drop unmapped 
dataframe=dataframe.dropna(subset=['label'])
print(f"\nAfter mapping: {dataframe.shape}")
print(f"\nClass distribution: \n{dataframe['label'].value_counts()}")

#replace 'unknown' string with actual NaN
dataframe['V_gene']=dataframe['V_gene'].replace('unknown', pd.NA)
dataframe['J_gene']= dataframe['J_gene'].replace('unknown', pd.NA)

dataframe=dataframe.dropna(subset=['CDR3'])
#sequence validation
validaa=set('ACDEFGHIKLMNPQRSTVWY')

def validsequence(seq):
    seq = str(seq).upper()
    for c in seq:
        if c not in validaa:
            return False
    return True

dataframe=dataframe[dataframe['CDR3'].apply(validsequence)]

print(f"\nFinal shape: {dataframe.shape}")
print(f"\nFinal class distribution:\n{dataframe['label'].value_counts()}")
print(f"\nMissing V_gene: {dataframe['V_gene'].isna().sum()}")
print(f"Missing J_gene: {dataframe['J_gene'].isna().sum()}")

#save
dataframe.to_csv('../data/TCR-cleaned.csv', index=False)
print("\nSaved to ../data/TCR-cleaned.csv")