# Winning solution proposed for the IFMBE 2019 Scientific Challenge

Source code of the winning solution based on Convolutional Neural Networks (CNNs) proposed for the IFMBE 2019 Scientific Challenge of the 15th Mediterranean Conference on Medical and Biological Engineering and Computing.

## Getting Started

### Prerequisites
* PyTorch
* NumPy

### Usage
The best-performing solution was based on an inter-session training strategy (i.e. developing participant-specific CNNs). 
The script "eegnet_bciaut_p300.py" contains the code to build the CNN architecture based on EEGNet used for the competition. 
The folder "attempt09_weights_and_prediction" contains the weights (file containing the model state dictionary "net_best.pth") and the predicted objects at each training epoch ("savers_epochs.csv") of the best inter-session attempt submitted during the phase II of the competition. 

## Citing
