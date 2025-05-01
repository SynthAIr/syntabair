[How to evaluate the quality of the synthetic data – measuring from the perspective of fidelity, utility, and privacy](https://aws.amazon.com/blogs/machine-learning/how-to-evaluate-the-quality-of-the-synthetic-data-measuring-from-the-perspective-of-fidelity-utility-and-privacy/)

The three dimensions of synthetic data quality evaluation: 

The synthetic data generated is measured against three key dimensions:

    Fidelity
    Utility
    Privacy

These are some of the questions about any generated synthetic data that should be answered by a synthetic data quality report:

    How similar is this synthetic data as compared to the original training set?
    How useful is this synthetic data for our downstream applications?
    Has any information been leaked from the original training data into the synthetic data?


## The different dimensions of synthetic data quality

### Fidelity
  The fidelity of synthetic data is determined by two factors: how closely the generated data matches the original data points and how much diversity is preserved in the newly created synthetic points.
Fidelity metrics assist us in determining how closely the generated data matches the original data, thus instilling confidence in its suitability for exploration and analysis.

### Utility
  The utility score measures how useful synthetic data can be in replacing real data for downstream applications, such as analytics or even machine learning models.
  
### Privacy
  Privacy indicates the level of confidentiality of synthetic data. When validating the privacy of generated synthetic data, it is important to classify the different bits of information according to the identification risk and value they hold for the downstream application. 

### Mapping of the SDMetrics “single-table” scores to the four high-level evaluation pillars

| Pillar | Metrics that belong here | Rationale |
|--------|-------------------------|-----------|
| **Fidelity**<br>*How closely does the synthetic table mimic the real table’s statistical structure?* | • **BNLikelihood**  <br>• **BNLogLikelihood**  <br>• **GMLogLikelihood**  <br>• **LogisticDetection**  <br>• **SVCDetection**  <br>• **CSTest**  <br>• **KSComplement**  <br>• **ContinuousKLDivergence**  <br>• **DiscreteKLDivergence** | Likelihood, KS, KL and chi-square tests measure overlap of the real and synthetic distributions (univariate or multivariate).<br>The two *Detection* metrics ask “can a classifier tell them apart?”, which is another fidelity lens. |
| **Utility**<br>*How well can models trained on synthetic data generalise to real-world data for the intended ML task?* | • **BinaryDecisionTreeClassifier**  <br>• **BinaryAdaBoostClassifier**  <br>• **BinaryLogisticRegression**  <br>• **BinaryMLPClassifier**  <br>• **MulticlassDecisionTreeClassifier**  <br>• **MulticlassMLPClassifier**  <br>• **LinearRegression**  <br>• **MLPRegressor**  <br>• **MLEfficacy** | These ML-Efficacy scores train a model **only on the synthetic rows** and report its performance on real rows, so they directly quantify downstream usefulness. |
| **Privacy**<br>*How hard is it for an attacker to infer secrets about the original data set?* | • **CategoricalCAP**  <br>• **CategoricalZeroCAP**  <br>• **CategoricalGeneralizedCAP**  <br>• **CategoricalKNN**  <br>• **CategoricalNB**  <br>• **CategoricalRF**  <br>• **CategoricalEnsemble**  <br>• **NumericalMLP**  <br>• **NumericalLR**  <br>• **NumericalSVR**  <br>• **NumericalRadiusNearestNeighbor** | All of these intentionally train an adversarial model on the synthetic data and report its success rate (or probability) when attacking real data. Lower success ⇒ stronger privacy. |




# 🧩 Mapping Metrics to Categories

## 1. **Fidelity (Statistical Similarity)**

Statistical Similarity (Data Fidelity)

This measures how closely the synthetic data mimics the real data's distributions, relationships, and statistical properties.

These metrics evaluate **how statistically similar** your synthetic data is to the real data.

| Metric | Description |
|:-------|:------------|
| **BNLikelihood** | How likely the synthetic data is under a Bayesian Network trained on real data. |
| **BNLogLikelihood** | Log-likelihood variant, same purpose as above. |
| **GMLogLikelihood** | Log-likelihood under a Gaussian Mixture model fitted to real data. |
| **CSTest** | Chi-Squared Test across categorical columns, compares distributions. |
| **KSComplement** | Kolmogorov-Smirnov test complement for numerical columns. |
| **ContinuousKLDivergence** | KL divergence between pairs of numerical columns. |
| **DiscreteKLDivergence** | KL divergence between pairs of discrete (categorical/boolean) columns. |


---

## 2. **Utility (Downstream ML Task Performance)**

Utility (Downstream Task Performance)

This tests if models trained on synthetic data are as useful as models trained on real data.

These metrics test **how useful** the synthetic data is for machine learning models.

| Metric | Description |
|:-------|:------------|
| **BinaryDecisionTreeClassifier** | Train decision tree on synthetic for binary classification. |
| **BinaryAdaBoostClassifier** | Train AdaBoost on synthetic for binary classification. |
| **BinaryLogisticRegression** | Train logistic regression on synthetic for binary classification. |
| **BinaryMLPClassifier** | Train MLP (neural network) on synthetic for binary classification. |
| **MulticlassDecisionTreeClassifier** | Train decision tree on synthetic for multiclass classification. |
| **MulticlassMLPClassifier** | Train MLP on synthetic for multiclass classification. |
| **LinearRegression** | Linear regression for regression tasks. |
| **MLPRegressor** | MLP for regression tasks. |
| **MLEfficacy** | A wrapper that detects task type automatically and applies appropriate metrics. |


---

## 3. **Privacy (Disclosure Risk)**
Privacy (Disclosure Risk)

Synthetic data should protect privacy — real individuals should not be easily re-identified.

These metrics assess **how vulnerable** the synthetic data is to privacy attacks.

| Metric | Description |
|:-------|:------------|
| **CategoricalCAP** | Correct Attribution Probability for categorical columns. |
| **CategoricalZeroCAP** | Variant of CAP for categorical columns. |
| **CategoricalGeneralizedCAP** | Generalized CAP metric. |
| **CategoricalKNN** | KNN model to guess real records. |
| **CategoricalNB** | Naive Bayes to guess real records. |
| **CategoricalRF** | Random Forest for attack simulation. |
| **CategoricalEnsemble** | Ensemble of models for stronger attacks. |
| **NumericalLR** | Linear Regression attack for numerical columns. |
| **NumericalMLP** | MLP attack for numerical columns. |
| **NumericalSVR** | Support Vector Regressor attack. |
| **NumericalRadiusNearestNeighbor** | Radius-based nearest neighbor attack on numerical data. |



---

## 4. **Diversity (Coverage of Distribution)**
Coverage and Diversity

This measures whether the synthetic data covers the full range of real-world scenarios, including rare cases.


However, **some Fidelity metrics can indirectly hint at diversity issues**, especially:
- **GMLogLikelihood**: If synthetic data collapses to a few modes, GMM log-likelihoods will be low.
- **ContinuousKLDivergence / DiscreteKLDivergence**: High KL divergence might signal missing modes.
- **CSTest** and **KSComplement**: If rare categories or tails of distributions are missing, these scores degrade.

---

## Detection

Tabular Detection describes a set of metrics that calculate how difficult it is to tell apart the real data from the synthetic data. 

This is done using a machine learning model. There are two different Detection metrics that use different ML algorithms: LogisticDetection and SVCDetection.



### How does it work?

- Create a single, augmented table that has all the rows of real data and all the rows of synthetic data. - Add an extra column to keep track of whether each original row is real or synthetic.
- Split the augmented data to create a training and validation sets.
- Choose a machine learning model based on the metric used (see below). Train the model on the training split. The model will predict whether each row is real or synthetic (ie predict the extra column we created in step #1)
- Validate the model on the validation set

### Score

The final score is based on the average ROC AUC score across all the cross validation splits.
$score=1−(max(ROC AUC,0.5)×2−1)$


(highest) 1.0: The machine learning model cannot identify the synthetic data apart from the real data

(lowest) 0.0: The machine learning model can perfectly identify synthetic data apart from the real data

A score of 1 may indicate high quality but it could also be a clue that the synthetic data is leaking privacy (for example, if the synthetic data is copying the rows in the real data).



##

```
src/embeddings/evaluation/
├── __init__.py
├── fidelity/
│   ├── __init__.py
│   ├── distribution.py    # Distributional similarity metrics (KL, JS divergence)
│   ├── detection.py       # Detection metrics (logistic, SVC)
│   ├── likelihood.py      # Bayesian and GMM likelihood metrics
│   ├── statistical.py     # Statistical tests (KS, CS tests)
│   └── correlation.py     # Correlation preservation metrics
├── utility/
│   ├── __init__.py
│   ├── tstr.py            # Your existing TSTR implementation
│   ├── ml_efficacy.py     # Machine Learning efficacy metrics
│   └── downstream_tasks.py # Performance on specific downstream tasks
├── privacy/
│   ├── __init__.py
│   ├── cap.py             # Correct Attribution Probability methods
│   ├── nn_attacks.py      # Nearest neighbor attacks
│   ├── ml_attacks.py      # ML-based privacy attacks
│   └── disclosure_risk.py # Disclosure risk assessment
├── diversity/
│   ├── __init__.py
│   ├── coverage.py        # Coverage of the original data space
│   ├── novelty.py         # Generation of novel but plausible samples
│   └── mode_collapse.py   # Detection of mode collapse issues
└── common/
    ├── __init__.py
    ├── metrics.py         # Base classes for metrics
    ├── data_preparation.py # Common data preparation utilities
    ├── reporting.py       # Reporting utilities
    └── visualization.py   # Visualization utilities
```