# INTRUSION DETECTION SYSTEM
 
 Overview
This repository implements a deep learning-based Network Intrusion Detection System (NIDS) that utilizes a hybrid architecture combining Convolutional Neural Networks (CNNs), Bidirectional LSTMs, and a custom Self-Attention mechanism. The model is designed to classify network traffic as either normal or malicious and can operate across multiple benchmark intrusion detection datasets.

Theoretical Background
Network Intrusion Detection
Network Intrusion Detection Systems (NIDS) monitor and analyze network traffic to identify signs of intrusions or malicious behavior. Types of attacks detected include:

Denial of Service (DoS): Disrupting legitimate access to systems

Probing: Scanning networks to discover vulnerabilities

User to Root (U2R): Privilege escalation by regular users

Remote to Local (R2L): Remote attackers gaining unauthorized local access


Deep Learning Approach
This system uses a hybrid deep learning architecture with the following core components:

1. Convolutional Neural Networks (CNNs)
Purpose: Learn local features and spatial hierarchies

Layer: Conv1D with 64 filters and kernel size of 3

Activation: ReLU

Benefit: Detects local dependencies in feature sequences

2. Bidirectional Long Short-Term Memory (BiLSTM)
Purpose: Capture long-term dependencies and sequential behavior

Implementation: Bidirectional LSTM with 64 units

Benefit: Processes input sequences in both forward and backward directions, enhancing context awareness

3. Custom Self-Attention Layer
Purpose: Assign dynamic importance to different parts of the sequence

Inspired By: Transformer architecture from Vaswani et al., 2017

Mechanism: Computes attention scores using learned query, key, and value vectors

Benefit: Provides model interpretability and emphasizes relevant features

Feature Processing
Label Encoding: Converts categorical features to numeric form

Missing & Infinite Values: Replaced using median imputation

Normalization: StandardScaler scales numerical values to mean 0 and std 1

Reshaping: Input reshaped to 3D (samples, features, channels) for CNNs

 Supported Datasets
1. NSL-KDD
Description: Cleaned version of KDD Cup '99 without redundancy

Features: 41 features capturing various aspects of connections

Classes: normal and multiple attack types

2. CICIDS2017
Description: Modern traffic patterns including DoS, DDoS, botnets, and more

Features: 80+ flow-based features extracted via CICFlowMeter

Classes: BENIGN vs various attack types

3. UNSW-NB15
Description: Comprehensive modern dataset of synthetic normal and attack traffic

Features: 49 statistical and protocol-based features

Classes: 0 = normal, 1 = attack

Model Architecture:

Input Layer
    ↓
1D Convolution (Conv1D)
    ↓
MaxPooling1D
    ↓
Bidirectional LSTM
    ↓
Self-Attention Layer
    ↓
GlobalAveragePooling1D
    ↓
Dense Layer (ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (Softmax)

Evaluation Metrics
The model is evaluated using the following metrics:

Accuracy-Overall prediction correctness
Precision-Fraction of true positives among predicted positives
Recall-Fraction of true positives among actual positives
F1 Score-Harmonic mean of precision and recall
AUC	Area-under the ROC curve (multi-class OVR)
FPR	False- Positive Rate
TNR	True- Negative Rate
