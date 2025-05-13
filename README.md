# Intrusion Detection using CNN-BiLSTM with Self-Attention

This project builds a deep learning model that combines Convolutional Neural Networks (CNNs), Bidirectional LSTM, and a custom Self-Attention mechanism to classify network traffic as either normal or attack on datasets like NSL-KDD, CICIDS2017, and UNSW-NB15.



## Objective

To develop a deep learning-based Intrusion Detection System (IDS) that generalizes across multiple datasets using a hybrid model composed of:

- 1D Convolution for feature extraction  
- Bidirectional LSTM for sequence modeling  
- Self-Attention for adaptive weighting of important features  

---

## Libraries Used

| Library     | Description                         |
|------------|-------------------------------------|
| pandas     | DataFrame handling                  |
| numpy      | Numerical operations                |
| tensorflow | Deep learning model implementation |
| sklearn    | Preprocessing, model selection, metrics |
| docx       | Optional: for extracting field names |

---

## Datasets Supported

### NSL-KDD
- Format: KDDTrain+.txt
- Binary classes: normal / attack

### CICIDS2017
- Format: combine.csv
- Binary classes: BENIGN → normal, all others → attack

### UNSW-NB15
- Format: UNSW-NB15_1.csv
- Binary classes: 0 → normal, 1 → attack

---

## Data Preprocessing

- Label Encoding of categorical features.
- Missing Value Handling: Convert infinities to NaN and fill with median.
- Standardization: Scale features using StandardScaler.
- Reshaping: Convert into a 3D shape suitable for Conv1D layers.
- Label Conversion: class column → binary labels → one-hot encoded.

---

## Model Architecture



Input → Conv1D → MaxPooling1D → BiLSTM → SelfAttention → GAP → Dense → Dropout → Output (Softmax)



### Key Components:

- *Conv1D*: Extracts local features from sequence data.
- *Bidirectional LSTM*: Captures temporal dependencies in both directions.
- *Custom Self-Attention Layer*: Weighs feature importance adaptively.
- *GlobalAveragePooling1D*: Reduces the sequence into a single vector.
- *Dropout*: Prevents overfitting.
- *Dense (Softmax)*: Outputs class probabilities.

---

## Evaluation Metrics

After training, the model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC (ROC)  
- Confusion Matrix  
- False Positive Rate (FPR)  
- True Negative Rate (TNR)  

---


