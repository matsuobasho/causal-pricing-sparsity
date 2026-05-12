# Causal Inference for Marketplace Elasticity
**Estimating Price Sensitivity in Sparse Geographic Environments using DML and Entity Embeddings**

## Project Overview
This project focuses on estimating the causal impact of price shocks (using tolls as a proxy) on rider behavior (tip percentages) within the NYC TLC (Taxi & Limousine Commission) dataset. 

The primary challenge addressed is **geographic sparsity**. Standard causal models (like OLS or basic DML) struggle with high-cardinality features like `LocationID` where many zones have low trip volume. This project implements a structural approach using **Double Machine Learning (DML)** and **Latent Entity Embeddings** to regularize treatment effects across the marketplace.

## Technical Stack
*   **Language:** Python 3.9+
*   **Causal Framework:** `EconML` (Double Machine Learning)
*   **Deep Learning:** `PyTorch` (Entity Embeddings for high-cardinality features)
*   **Data Processing:** `Pandas`, `NumPy`, `Scikit-Learn`

## Methodology

### 1. Causal Identification
We define the relationship using the partially linear model:
- **Treatment ($T$):** Binary indicator of whether a trip incurred a toll (Proxy for price shock).
- **Outcome ($Y$):** Tip percentage of the total fare.
- **Confounders ($X, W$):** Time of day, day of week, trip distance, and LocationIDs.

### 2. Addressing Sparsity via Embeddings
Instead of one-hot encoding 260+ `LocationIDs`, we utilize a **PyTorch-based embedding layer**. This maps geographic zones into a lower-dimensional latent space, allowing the model to:
*   Share information between "similar" neighborhoods.
*   Reduce the variance of treatment effect estimates in sparse, low-volume zones.
*   Prevent overfitting in the nuisance models ($M(X, W)$ and $G(X, W)$).

### 3. Double Machine Learning (DML)
We employ a **Cross-Fit** procedure to avoid over-fitting:
*   **Stage 1:** Train two models to predict $T$ and $Y$ from confounders $X$.
*   **Stage 2:** Residualize $T$ and $Y$ (remove the influence of confounders).
*   **Stage 3:** Perform a final regression on residuals to identify the **Conditional Average Treatment Effect (CATE)**.

## Repository Structure
```text
├── notebooks/
│   └── eda_and_embedding_init.ipynb   # Exploratory analysis & spatial sparsity audit
├── src/
│   ├── data_loader.py                 # NYC TLC API / Parquet ingestion
│   ├── embeddings.py                  # PyTorch Entity Embedding architecture
│   └── model_dml.py                   # EconML wrapper for Causal Forest/DML
├── requirements.txt
└── README.md
