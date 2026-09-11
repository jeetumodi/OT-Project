# Neural-XGBoost (N-XGB) Disaster Prediction: Comprehensive Technical & Empirical Report

---

## 1. Executive Summary & Research Background

This document provides a complete technical specification, pipeline breakdown, mathematical formulation, and empirical evaluation of the Neural-XGBoost (N-XGB) hybrid architecture implemented in [`nxgb_classifier.py`](./nxgb_classifier.py).

### 1.1 Paper Reference
- **Title**: *Neural-XGBoost: A Hybrid Approach for Disaster Prediction and Management Using Machine Learning*
- **Journal**: IEEE Access (2025)
- **Problem Statement**: Natural disaster datasets (EM-DAT) present severe class imbalances, high-dimensional non-linear correlations, and missing values. Traditional models struggle with either tabular feature representation or minority class sensitivity.
- **Proposed Solution**: A hybrid deep learning and gradient boosted tree framework (N-XGB) where a multi-layer Neural Network acts as a non-linear latent feature extractor ($H \in \mathbb{R}^{32}$), coupled with SMOTE latent space balancing and XGBoost classification.

---

## 2. End-to-End Pipeline Architecture

```
+-----------------------------------------------------------------------------------+
|                              EM-DAT GLOBAL DATASET                                |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|  STAGE 1: DATA PREPROCESSING & DATA LEAKAGE PREVENTION                            |
|  - Filter to 3 classes: Wildfire, Flood, Earthquake                               |
|  - Drop identifier & target-leaking taxonomic columns                             |
|  - Median Imputation (Numerical) & Mode Imputation (Categorical)                  |
|  - Label Encoding (Categorical)                                                   |
|  - Stratified 70:30 Split (Train = 3,738 | Test = 1,602)                         |
|  - StandardScaler fitted strictly on Train split                                  |
+-----------------------------------------------------------------------------------+
                            |                                    |
                            v                                    v
+---------------------------------------+          +--------------------------------+
| STAGE 2: PYTORCH FEATURE EXTRACTOR    |          | Holdout Test Set Transformed   |
| Architecture: Input -> 128 -> 64 -> 32|          | by Scaler & Trained NN Weights |
| Output: 32-dim Latent Embeddings (H)  |          +--------------------------------+
+---------------------------------------+                          |
         |                     |                                   |
         v                     v                                   |
   [Raw Features X]    [Latent Features H]                         |
         |                     |                                   |
         +----------+----------+                                   |
                    |                                              |
                    v                                              |
+---------------------------------------+                          |
| STAGE 3: CLASS BALANCING (SMOTE)      |                          |
| Balanced (SMOTE) vs Imbalanced (Raw)  |                          |
+---------------------------------------+                          |
                    |                                              |
                    v                                              v
+---------------------------------------+          +--------------------------------+
| STAGE 4: 5-FOLD STRATIFIED GRIDSEARCH | -------> | STAGE 5: COMPREHENSIVE METRICS |
| Models: LR, SVM, RF, k-NN, N-XGB      |          | - Confusion Matrices           |
+---------------------------------------+          | - Per-Class & Macro ROC-AUC    |
                                                   | - Precision, Recall, Spec, F1  |
                                                   +--------------------------------+
```

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Feature Extraction Network (`FeatureExtractorNN`)
The neural network compresses the $N$-dimensional input space down to a 32-dimensional latent feature representation $H$:

$$\mathbf{z}_1 = \text{ReLU}\left(\text{BatchNorm}\left(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1\right)\right), \quad \mathbf{z}_1 \in \mathbb{R}^{128}, \quad \text{Dropout}(p=0.3)$$

$$\mathbf{z}_2 = \text{ReLU}\left(\text{BatchNorm}\left(\mathbf{W}_2 \mathbf{z}_1 + \mathbf{b}_2\right)\right), \quad \mathbf{z}_2 \in \mathbb{R}^{64}, \quad \text{Dropout}(p=0.3)$$

$$\mathbf{h} = \text{ReLU}\left(\text{BatchNorm}\left(\mathbf{W}_3 \mathbf{z}_2 + \mathbf{b}_3\right)\right), \quad \mathbf{h} \in \mathbb{R}^{32}$$

$$\hat{\mathbf{y}} = \text{Softmax}\left(\mathbf{W}_4 \mathbf{h} + \mathbf{b}_4\right), \quad \hat{\mathbf{y}} \in \mathbb{R}^{3}$$

### 3.2 SMOTE Resampling in Latent Space
For any minority class sample $\mathbf{h}_i \in H_{\text{train}}$, a synthetic instance $\mathbf{h}_{\text{new}}$ is generated by selecting a $k$-nearest neighbor $\mathbf{h}_{zi}$ and interpolating with random factor $\lambda \in [0, 1]$:

$$\mathbf{h}_{\text{new}} = \mathbf{h}_i + \lambda \cdot (\mathbf{h}_{zi} - \mathbf{h}_i)$$

### 3.3 XGBoost Objective on Latent Representation $H$
The objective function minimized across the boosted tree ensemble:

$$\mathcal{L}(\theta) = \sum_{i=1}^{M} l\left(y_i, \hat{y}_i^{(t)}\right) + \sum_{k=1}^K \Omega(f_k)$$

$$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

---

## 4. Hyperparameter Search Grids

| Model | Tuned Hyperparameters | Search Space |
| :--- | :--- | :--- |
| **Logistic Regression** | Regularization strength $C$<br>Optimization Solver | $C \in [0.01, 0.1, 1, 10]$<br>`solver` $\in [\text{'lbfgs'}, \text{'saga'}]$ |
| **SVM** | Penalty parameter $C$<br>Kernel type<br>Kernel coefficient $\gamma$ | $C \in [0.1, 1, 10]$<br>`kernel` $\in [\text{'rbf'}, \text{'linear'}]$<br>$\gamma \in [\text{'scale'}, \text{'auto'}]$ |
| **Random Forest** | Number of Trees ($n\_estimators$)<br>Max Tree Depth<br>Min Samples Split | $n\_estimators \in [100, 200, 300]$<br>$max\_depth \in [5, 7, 10]$<br>$min\_samples\_split \in [2, 5]$ |
| **k-NN** | Number of Neighbors ($k$)<br>Weight Function<br>Distance Metric | $k \in [3, 5, 7, 9, 11]$<br>`weights` $\in [\text{'uniform'}, \text{'distance'}]$<br>`metric` $\in [\text{'euclidean'}, \text{'manhattan'}]$ |
| **Proposed N-XGB** | Learning rate $\eta$<br>Max Tree Depth<br>Number of Estimators | $\eta \in [0.01, 0.05, 0.1]$<br>$max\_depth \in [5, 7, 9]$<br>$n\_estimators \in [100, 200, 300]$ |

---

## 5. Visual Artifact & Performance Analysis

### 5.1 Dataset Partitioning & Class Distribution
Evaluation is carried out on an independent $30\%$ holdout test partition containing **$1{,}602$ samples**:
- **Flood** (Majority): $1{,}286$ samples ($80.27\%$)
- **Wildfire** (Minority): $212$ samples ($13.23\%$)
- **Earthquake** (Severe Minority): $104$ samples ($6.49\%$)

---

### 5.2 Confusion Matrix Breakdown

#### A. Balanced Training (SMOTE) — `cm_balanced.png`
![Confusion Matrices - Balanced](./cm_balanced.png)

*Detailed Confusion Matrix Cell Counts ($N_{\text{test}} = 1{,}602$):*

| Metric / Class | Logistic Regression | SVM | Random Forest | k-NN | Proposed N-XGB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Wildfire TP (out of 212)** | 208 ($98.11\%$) | 184 ($86.79\%$) | **211 ($99.53\%$)** | 160 ($75.47\%$) | 201 ($94.81\%$) |
| **Flood TP (out of 1286)** | 901 ($70.06\%$) | 1,064 ($82.74\%$) | **1,213 ($94.32\%$)** | 1,053 ($81.88\%$) | 1,198 ($93.16\%$) |
| **Earthquake TP (out of 104)** | **63 ($60.58\%$)** | 57 ($54.81\%$) | 59 ($56.73\%$) | 60 ($57.69\%$) | 37 ($35.58\%$) |
| **Total Correct Predictions** | 1,172 / 1,602 | 1,305 / 1,602 | **1,483 / 1,602** | 1,273 / 1,602 | 1,436 / 1,602 |
| **Overall Accuracy** | **$73.16\%$** | **$81.46\%$** | **$92.57\%$** | **$79.46\%$** | **$89.64\%$** |

---

#### B. Imbalanced Training (Raw) — `cm_imbalanced.png`
![Confusion Matrices - Imbalanced](./cm_imbalanced.png)

*Detailed Confusion Matrix Cell Counts ($N_{\text{test}} = 1{,}602$):*

| Metric / Class | Logistic Regression | SVM | Random Forest | k-NN | Proposed N-XGB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Wildfire TP (out of 212)** | 137 ($64.62\%$) | 198 ($93.40\%$) | **211 ($99.53\%$)** | 110 ($51.89\%$) | 194 ($91.51\%$) |
| **Flood TP (out of 1286)** | 1,275 ($99.14\%$) | 1,230 ($95.65\%$) | **1,283 ($99.77\%$)** | 1,232 ($95.80\%$) | 1,251 ($97.28\%$) |
| **Earthquake TP (out of 104)** | 4 ($3.85\%$) | 0 ($0.00\%$) | 21 ($20.19\%$) | **24 ($23.08\%$)** | 22 ($21.15\%$) |
| **Total Correct Predictions** | 1,416 / 1,602 | 1,428 / 1,602 | **1,515 / 1,602** | 1,366 / 1,602 | 1,467 / 1,602 |
| **Overall Accuracy** | **$88.39\%$** | **$89.14\%$** | **$94.57\%$** | **$85.27\%$** | **$91.57\%$** |

---

### 5.3 Multi-Model ROC-AUC Comparison

| Model | Balanced ROC-AUC (`roc_balanced.png`) | Imbalanced ROC-AUC (`roc_imbalanced.png`) |
| :--- | :---: | :---: |
| **Random Forest** | **0.95843** | **0.96251** |
| **Proposed N-XGB (Hybrid)** | **0.91082** | **0.91535** |
| **SVM** | **0.90270** | **0.83327** |
| **Logistic Regression** | **0.88009** | **0.87072** |
| **k-NN** | **0.83879** | **0.81448** |

#### Combined ROC Curves:
| Balanced ROC Comparison | Imbalanced ROC Comparison |
| :---: | :---: |
| ![ROC Balanced](./roc_balanced.png) | ![ROC Imbalanced](./roc_imbalanced.png) |

---

### 5.4 Neural-XGBoost Per-Class ROC-AUC Curves

| Disaster Class | Balanced AUC (`roc_nxgb_balanced.png`) | Imbalanced AUC (`roc_nxgb_imbalanced.png`) |
| :--- | :---: | :---: |
| **Wildfire** | **0.99161** | **0.98704** |
| **Flood** | **0.91779** | **0.92304** |
| **Earthquake** | **0.82307** | **0.83598** |

| N-XGB ROC on Balanced Data | N-XGB ROC on Imbalanced Data |
| :---: | :---: |
| ![N-XGB ROC Balanced](./roc_nxgb_balanced.png) | ![N-XGB ROC Imbalanced](./roc_nxgb_imbalanced.png) |

---

## 6. Key Scientific & Operational Insights

1. **The Majority-Class Illusion**:
   - In the imbalanced scenario, raw accuracy is deceptively high ($\sim 89-94\%$), because predicting the majority class (`Flood`) dominates the score.
   - Without SMOTE, SVM completely fails on Earthquakes ($0\%$ sensitivity, $0$ true positives).

2. **Impact of Latent SMOTE**:
   - Resampling in the 32-dimensional latent embedding space ($H$) provides smooth synthetic feature vectors, preventing standard high-dimensional distortion.
   - Earthquake recall dramatically increases from $21.15\%$ (imbalanced) to $35.58\%$ (balanced) in N-XGB, and up to $60.58\%$ in Logistic Regression.

3. **High Discriminative Power of N-XGB**:
   - The hybrid architecture delivers near-perfect discrimination for Wildfires ($\text{AUC} > 0.987$) and strong performance for Floods ($\text{AUC} > 0.917$), achieving an overall ROC-AUC of $\mathbf{0.91535}$ and an accuracy of $\mathbf{91.57\%}$.

---

## 7. Instructions for Execution

To reproduce all benchmarks, tables, and visualization artifacts:

```bash
# Activate virtual environment if available
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Run complete pipeline
python nxgb_classifier.py
```
