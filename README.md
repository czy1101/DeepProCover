

## Install dependencies

```python
numpy
pandas
scipy
lightgbm
networkx
matplotlib
copy
time
joblib
bayes_opt
triqler
scikit-learn
torch
argparse
warnings
```



This repository contains scripts for processing mass spectrometry data, extracting features, performing feature selection, and rescoring PSMs using machine learning models.



## Description of Each File

### data_util.py

Contains utility functions for data loading, normalization, and basic preprocessing.



### SampleRT.py

Predicts retention time (RT)-related features such as predicted_RT and deltaRT, based on trained RT prediction models.



### MambaConvBiLSTM.py

Extracts features related to fragment ion intensities using a hybrid architecture combining Mamba and BiLSTM. Outputs include similarity-based metrics such as Cosine and PEPCosine.



### RL.py

Performs reinforcement learning-based feature selection to identify the most informative subset of features for improving model performance and interpretability.



### LgbBayes_house.py / LgbBayes_N2.py / LgbBayes_nanowell.py

Apply LightGBM with Bayesian optimization to rescore PSMs in different experimental settings (house, N2, and nanowell). Choose the appropriate script based on your dataset.



### PSMsFilter.py

Filters PSMs based on quality control criteria such as Score, Charge, Length, PIF, and Intensity.






## Suggested Workflow

Data Preparation & Filtering: data_util.py, PSMsFilter.py



Feature Generation: SampleRT.py, MambaConvBiLSTM.py



Feature Selection: RL.py



Rescoring: Use the appropriate LgbBayes_*.py script depending on your experiment.

 