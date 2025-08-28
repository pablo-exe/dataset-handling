# Machine Learning on Classic Datasets

Three end-to-end notebooks exploring regression and classification on well-known tabular datasets. Each notebook follows a complete workflow: exploratory analysis, preprocessing, model design, training, evaluation, and visualization.

---

## Repository Structure

- `1. Dataset Diabetes.ipynb` — Regression on the Diabetes dataset (continuous target).
- `2. Dataset Boston. Dataset Boston. Dataset Boston.ipynb` — Regression on the Boston Housing dataset (legacy/educational).
- `4. Dataset Cancer Breast.ipynb` — Binary classification on the Breast Cancer Wisconsin dataset.

---

## Notebook Summaries

### 1) `1. Dataset Diabetes.ipynb` — Regression (Diabetes)
**Dataset:** 442 samples × 10 features (`sklearn.datasets.load_diabetes`).  
**Exploration:**
- Feature line plots (e.g., **BMI / Mass Index**, **Blood Pressure**) across samples.
- Scatter plots showing relationships (e.g., **BMI** vs **Blood Pressure**, **Age** vs **Blood Pressure**).

**Modeling:**
- Train/test split: **80/20**, `random_state=42`.
- **Keras** feed-forward network:
    - Dense(25, activation=`relu`) → Dense(1)
    - Loss: **MeanSquaredError**; Metric: **MAE**; Optimizer: **Adam**
    - **50 epochs**, **batch_size=4**
- Learning-curve plots for **training/validation loss** and **MAE**.

**Key takeaways:**
- End-to-end regression pipeline without explicit scaling to illustrate baseline behavior.
- Interpretation of learning curves to reason about capacity and generalization.

---

### 2) `2. Dataset Boston. Dataset Boston. Dataset Boston.ipynb` — Regression (Boston Housing)
**Dataset:** 506 samples × 13 features (`sklearn.datasets.load_boston`, legacy).  
**Exploration:**
- Feature trends (e.g., **Industrial proportion**, **Distance to employment centers**, **Rooms**).
- Summary statistics (per-feature **mean** and **variance**).
- **3D scatter**: `Distance × Rooms → Price` for multi-feature/target relationships.

**Preprocessing:**
- **MinMaxScaler** on all features to [0, 1].

**Modeling:**
- Train/test split: **80/20**, `random_state=42`.
- **Keras** feed-forward network:
    - Dense(25, activation=`relu`) → Dense(1)
    - Loss: **MeanSquaredError**; Metric: **MAE**; Optimizer: **Adam**
    - **60 epochs**, **batch_size=4**
- Learning-curve plots for **training/validation loss** and **MAE**.

**Key takeaways:**
- Demonstrates how **feature scaling** stabilizes optimization in neural regression.
- Shows a compact network capturing non-linear relationships in tabular data.

> Note: The Boston dataset is deprecated in recent scikit-learn versions and is used here strictly for educational purposes.

---

### 3) `4. Dataset Cancer Breast.ipynb` — Classification (Breast Cancer)
**Dataset:** 569 samples × 30 features (`sklearn.datasets.load_breast_cancer`).  
**Exploration:**
- Feature trends (e.g., **Perimeter**, **Area**) and pairwise **scatter** diagnostics.

**Preprocessing:**
- **MinMaxScaler** on features.

**Modeling:**
- Train/test split: **80/20**, `random_state=42`.
- **Keras** feed-forward classifier:
    - Dense(25, activation=`relu`) → Dense(1, activation=`sigmoid`)
    - Loss: **binary_crossentropy**; Metric: **accuracy**; Optimizer: **Adam**
    - **50 epochs**, **batch_size=4**
- Learning-curve plots for **training/validation loss** and the chosen metric.

**Key takeaways:**
- Compact neural classifier for **benign vs malignant** prediction.
- Shows a minimal, interpretable baseline suitable for feature-importance follow-ups.

---

## Techniques & Skills Demonstrated

- **Exploratory Data Analysis:** line/scatter plots, summary stats, 3D visualization for multivariate intuition.
- **Preprocessing:** train/test split, **MinMax scaling** (where needed), reproducible pipelines (`random_state=42`).
- **Neural Modeling (Keras):** shallow MLPs for tabular **regression** and **binary classification**.
- **Optimization & Training:** Adam optimizer, small batch sizes to highlight gradient dynamics on small datasets.
- **Evaluation:** **MSE/MAE** for regression, **accuracy** for classification; training vs validation curves to assess over/underfitting.
- **Good Practices:** modular code cells, clear separation of EDA → preprocessing → modeling → evaluation.

---

## Environment

**Install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn keras tensorflow
Run notebooks: jupyter notebook
