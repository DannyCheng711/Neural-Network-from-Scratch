# Neural Network from Scratch

This project contains two parts:  

1. **Part 1 - NN library from scratch (NumPy only):** Implements core neural network components — linear layers, activation and loss functions, forward and backward propagation, mini-batch SGD updates, a simple training loop, and data preprocessing. 
2. **Part 2 - House value regression:** Applies the implemented NN library to predict median house prices in California using a fully connected neural network, including preprocessing, model training, and evaluation.

## Project Structure

```
.
├── part1_nn_lib.py                    # NumPy-only NN library + example_main()
├── part2_house_value_regression.py    # Uses the library for regression
├── iris.dat                           # Example dataset for Part 1 (Iris, one-hot labels)
├── requirements.txt
└── README.md
```

## How to Run

```bash
python part1_nn_lib.py
python part2_house_value_regression.py
```

Each script will automatically execute the `example_main()` function.
 

## Part1: NN library from scratch

This file includes four main classes `Layer`, `MultiLayerNetwork`, `Trainer` and `Preprocessor`. 



1. **Layer:** Implements the basic building blocks, including:
    - **LinearLayer:** affine transformation with Xavier initialization
	- **Activation layers:** SigmoidLayer, ReluLayer
	- **Loss layers:** MSELossLayer, CrossEntropyLossLayer

2. **MultiLayerNetwork:**  Builds a multi-layer perceptron by stacking linear and activation layers.
Supports full forward propagation, backward propagation, and parameter updates for all layers. 

3. **Trainer:** Defines the training process, including:
	- Epoch iterations and mini-batch division
	- Forward → Loss → Backward → Parameter update loop
	- Optional dataset shuffling per epoch
	- Optimization method: Mini-batch Stochastic Gradient Descent (SGD)

4. **Preprocessor:**
Handles data normalization (min–max scaling) and supports both applying and reverting the transformation for evaluation or visualization.

## Part 2: House value regression

This part demonstrates how to use the pytorch NN library to predict median house prices in California.
It includes data preprocessing, neural network architecture design, training, and evaluation.

### Model Architecture

| Component | Description | 
| --- | --- |
| Input Layer | Number of neurons = number of input features after preprocessing |
| Hidden Layers |  2–5 fully connected layers (default: 2) with 64–256 neurons each |
| Activation | ReLU after each hidden layer (Mitigates vanishing gradients and speeds up training) |
Dropout | 20% rate to prevent overfitting |
| Output Layer |Single neuron with linear activation for regression | 
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam (adaptive learning rate) | 




### Data Preprocessing 

1. **Handling Missing Values:** Column total_bedrooms imputed using median (robust against skewed distributions).
2. **Encoding Categorical Variables:** ocean_proximity one-hot encoded.
3. **Normalization:** Numerical features standardized using Z-score normalization.

### Evaluation Metrics: 

1. **MSE:** Average squared difference between actual and predicted values
2. **RMSE:** Square root of MSE (in same units as price)
3. **R² Score:** Proportion of variance explained by model
4. **Adjusted R²:** R² adjusted for number of predictors


### Model Selection:

To identify the best-performing network configuration, we conducted hyperparameter tuning using the `Ray` library for distributed search.

The tuning process explored combinations of the following hyperparameters:


| Hyperparameter | Search Space | 
| --- | --- |
| Hidden layer size | {32, 64, 128, 256} |
| Number of hidden layers | {2, 3, 4, 5} |
| Learning rate | {0.01, 0.005, 0.001} | 
| Batch size | {64, 128} | 
| Dropout rate | {0.2, 0.3, 0.4} | 
| Epochs | {1000, 2000} | 

### Search Strategy:
- Each trial trained a full model on a 90% training subset and evaluated it on a 10% validation subset.
- The objective metric was Mean Squared Error (MSE) on the validation set.
- Ray Tune coordinated multiple experiments in parallel, automatically reporting intermediate results and selecting the best configurations.

    | Best Hyperparameter Found | Value | 
    | --- | --- |
    | Hidden layer size |  256 |
    | Number of hidden layers | 4 |
    | Learning rate | 0.001 |
    | Batch size | 64 |
    | Dropout rate | 0.2 |
    | Epochs | 2000 |



### Experimental Results
- Increasing hidden layer size improved validation RMSE and R² up to 256 neurons.
- Adding too many layers (5+) led to overfitting, where training loss decreased but validation loss increased.
- The 4-layer, 256-neuron model achieved the best trade-off between accuracy and generalization.
- Error distribution was approximately normal, centered near zero — no major bias detected.

---

### Acknowledgement

- **Program:** MSc Computing, Imperial College London
- **Module:** Machine Learning
- **Assignment:** Neural Network from Scratch