# TCRclass - TCR Source Classification Project

A 4-class TCR source classifier using ESM-2 and an attention-based classifier head.

**Competition result: Macro F1 = 0.40**

## Approach

CDR3β sequences are embedded using ESM-2 650M parameter protein language model. The resulting 1280 dimensional embeddings are passed through a multi-head attention classifier head that learns to distinguish between four TCR source classes. viral, bacterial, cancer, and autoimmune.

## REPO STRUCTURE
```bash
TCRclass/
├── data/
│   ├── TCR-Processed-Raw.csv      
│   └── test_set.csv               
├── src/
│   ├── clean.py                   
│   ├── embeddings.py              
│   ├── model.py                  
│   ├── train.py                   
│   ├── prediction.py              
│   ├─ attention.py
│   └─ input.py             
├── outputs/
│   ├── embeddings/               
│   ├── figures/                
│   └── model.pt                   
├── .gitignore
└── README.md
```
## Installation
```bash
git clone https://github.com/JitheshMithra/TCRclass.git
cd TCRclass
# Install PyTorch with CUDA first (required before other packages):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Then install remaining dependencies:
pip install -r requirements.txt

```

GPU strongly recommended. 
## How to Run

Run all scripts from the `src/` directory:

```bash
cd src
```

### Data Cleaning

```bash
python clean.py
```

Maps 31 raw pathology labels to 4 target classes, validates CDR3 sequences, handles missing V/J genes. Outputs `data/TCR-cleaned.csv`.

### Generate ESM-2 Embeddings

```bash
python embeddings.py
```

Runs all CDR3 sequences through ESM-2 650M and saves embeddings. 

### Step 3 Train the Model

```bash
python train.py
```

Trains the attention classifier on ESM-2 embeddings. Generates loss curves, confusion matrix, ROC curves, and Macro F1 validation curves.

Outputs `outputs/model.pt` and figures in `outputs/figures/`.

### Step 4 Generate Predictions

```bash
python prediction.py
```

You can run the trained model on a custom TCR sequence directly from the terminal:
```bash
python input.py --cdr3 CASSLAPGATNEKLFF --vgene TRBV12-3 --jgene TRBJ2-7
```
Arguments:
  - --cdr3 (required): CDR3β amino acid sequence
  - --vgene (optional): TRBV gene (default = unknown)
  - --jgene (optional): TRBJ gene (default = unknown)

The script will output predicted class probabilities and the final classification.

Additional Example (Without genes):
```bash
python input.py --cdr3 CASSIRSSYEQYF
```
Outputs `outputs/submission.csv` in csv format.

## Results

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Viral | 0.86 | 0.46 | 0.60 | 0.69 |
| Bacterial | 0.28 | 0.65 | 0.39 | 0.82 |
| Cancer | 0.22 | 0.50 | 0.30 | 0.74 |
| Autoimmune | 0.09 | 0.26 | 0.13 | 0.60 |
| **Macro avg** | **0.36** | **0.47** | **0.36** | |
| **Kaggle score** | | | **0.40** | |

## Output Figures:
<img width="1200" height="400" alt="training_curves" src="https://github.com/user-attachments/assets/d20e9cee-e577-4f77-8957-6e5030af9d2c" />
<img width="800" height="600" alt="roc_curves" src="https://github.com/user-attachments/assets/c02e5586-154c-4c2b-9dd8-92ebd578acf2" />
<img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/07aa4cf3-a532-42ca-b13c-c6f1f805aa8b" />
<img width="2100" height="1500" alt="attention_heatmaps" src="https://github.com/user-attachments/assets/bb9b6172-16e8-40c3-ae7d-1ec09f65d6e2" />


## Limitations and Assumptions

**Class imbalance:** Training data is mostly viral which creates heavily skewed decision despite class weighting. The model struggles most with autoimmune due to overlap with normal TCR repertoire diversity.

**Sequence only mode:** V/J gene features were not used in the final model. TRBV and TRBJ noise made clean label encoding unreliable. This is a valid improvement.

**Mean pooling limitation:** ESM-2 embeddings are mean-pooled over sequence length before the attention layer, losing positional information. Attention heatmaps are uniform and not biologically interpretable. Per-token embeddings would mean a full pipeline redesign.

**Single model:** Ensemble training was attempted but produced worse results due to insufficient architectural diversity between models.

**Dataset scope:** Only M. tuberculosis represents the bacterial class. The model may not generalize to other bacterial pathogens.

## Future Improvements

- V/J gene label encoding concatenated to ESM-2 embeddings
- Fine-tune last 2-3 ESM-2 transformer layers on TCR data
- Per-token embeddings to use meaningful attention visualization
- Cross-fold validation for more stronger performance estimates
- Architecturally diverse ensemble (V/J model + sequence-only model)

## Reproducibility

To exactly reproduce the competition submission:

```bash
python clean.py
python embeddings.py
python train.py    
python prediction.py
```

OPTIONAL ATTENTION HEATMAP GENERATION:
```bash
python attention.py
```
Hardware: NVIDIA RTX 4060, CUDA 13.1 driver
