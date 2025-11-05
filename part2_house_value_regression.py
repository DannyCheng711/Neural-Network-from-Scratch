import os
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# import ray
# import ray.tune as tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
from itertools import product
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score

USE_GPU = False  # GPU Setting

if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Using CPU only")

class Regressor(nn.Module): # Inherits nn.Module to allow PyTorch to manage the model

    def __init__(self, x, nb_epoch=1000, hidden_size=64, num_hidden_layers=2, learning_rate=0.001,  dropout_rate=0.2):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model with Dropout & L2 Regularization.
        
        Arguments:
            - x {pd.DataFrame} -- Raw input data to compute input size.
            - nb_epoch {int} -- Number of training epochs.
            - hidden_size {int} -- Number of neurons in each hidden layer.
            - num_hidden_layers {int} -- Number of hidden layers.
            - learning_rate {float} -- Learning rate for optimization.
            - dropout_rate {float} -- Dropout probability (default 0.3).
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        super().__init__()  # Ensure the initialization of nn.Module is executed

        # Process input data
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]  # Get the number of input features
        self.output_size = 1  # Predicting house prices, so there is a single output
        self.nb_epoch = nb_epoch  

        # Define Neural Network Architecture Dynamically
        layers = [nn.Linear(self.input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]  # First hidden layer

        for _ in range(num_hidden_layers - 1):  # Add additional hidden layers
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_size, self.output_size))  # Output layer

        self.model = nn.Sequential(*layers).to("cpu")  # Convert list to nn.Sequential

        # === Define Loss Function and Optimizer ===
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # Using Adam optimizer and L2
        
        return
    
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # === Handle missing values ===
        if training:
            self.median_values = x.select_dtypes(include=['number']).median()  # Store mean values (a series type) ONLY for numerical features
            self.fill_values = x.mode().iloc[0]  # Store mode for categorical features
        x = x.copy()
        x.fillna(self.median_values, inplace=True)  # Fill numerical missing values
        x.fillna(self.fill_values, inplace=True)  # Fill categorical missing values

        # === One-hot encode categorical variable: 'ocean_proximity' ===
        if training:
            self.label_binarizer = LabelBinarizer()  # Create binarizer
            ocean_encoded = self.label_binarizer.fit_transform(x['ocean_proximity']) # Only fit when training to set the encoding method (e.g. the number of categories)
        else:
            ocean_encoded = self.label_binarizer.transform(x['ocean_proximity'])  #  Apply transformation without re-fitting

        # Remove the original categorical column and append encoded columns
        x = x.drop(columns=['ocean_proximity'])
        ocean_encoded_df = pd.DataFrame(ocean_encoded, index=x.index)
        x = pd.concat([x, ocean_encoded_df], axis=1)

        # === Normalize numerical features ===
        numerical_cols = x.columns  # Collect all the features after adjusting categorical features
        if training:
            self.means = x[numerical_cols].mean() # Calculate mean for specified columns
            self.stds = x[numerical_cols].std()
        x[numerical_cols] = (x[numerical_cols] - self.means) / self.stds  # Apply normalization

        # Convert to torch tensors
        x_tensor = torch.tensor(x.values, dtype=torch.float32).to("cpu") # Tensor() can only include ndarray (x.values)
        # Process target variable y
        if y is not None:
            y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to("cpu") # Use view reshape to (batch_size, 1)
        else:
            y_tensor = None

        return x_tensor, y_tensor

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, batch_size=32):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Convert data to tensors
        X, Y = self._preprocessor(x, y=y, training=True)  # Preprocess data
        
        # Number of training samples
        num_samples = X.shape[0] 
        self.loss_history = [] # Store loss for every epoch

        for epoch in range(self.nb_epoch):
            # Shuffle data before each epoch to improve generalization
            indices = torch.randperm(num_samples)  
            X, Y = X[indices], Y[indices]
            
            epoch_loss = 0  # Track epoch loss
            
            for i in range(0, num_samples, batch_size):  # Iterate over batches
                X_batch = X[i:min(i + batch_size, num_samples)].to("cpu")
                Y_batch = Y[i:min(i + batch_size, num_samples)].to("cpu")
                
                # === Forward Pass ===
                predictions = self.model(X_batch)  # Compute predictions
                
                # === Compute Loss ===
                loss = self.loss_fn(predictions, Y_batch)  # Calculate MSE loss
                
                # === Backward Pass ===
                self.optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Compute gradients with respective to each weight
                self.optimizer.step()  # Update weights through Adam optimizer
                
                epoch_loss += loss.item()  # Accumulate batch loss 

            avg_loss = epoch_loss / (num_samples // batch_size + (num_samples % batch_size != 0))
            self.loss_history.append(avg_loss)  # Record Loss

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.nb_epoch}, Loss: {avg_loss:.4f}")

        return self  # Return trained model

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # === Preprocess the input data ===
        X, _ = self._preprocessor(x, training = False) # Preprocess the input data (training=False ensures no new fitting)

        # === Perform forward pass to get predictions ===
        with torch.no_grad():  # Disable gradient computation for inference to save computational consumption
            y_pred = self.model(X)
        
        # === Convert predictions from tensor to NumPy array ===
        return y_pred.numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y, plot=False):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # === Preprocess the input data ===
        X, Y = self._preprocessor(x, y = y, training = False)
        
        # === Preprocess the input data ===
        y_pred = self.predict(x)

        # ===  Compute evaluation metrics ===
        mse = mean_squared_error(Y.numpy(), y_pred)  # Mean Squared Error (MSE)
        rmse = np.sqrt(mse)  # Root Mean Squared Error (RMSE)
        r2 = r2_score(Y.numpy(), y_pred)  # Compute R² score
        
        # Compute Adjusted R²
        n = X.shape[0]  # Number of samples
        p = X.shape[1]  # Number of features
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")  # Prevent division error
        
        # === Print results ===
        print(f"Model Evaluation:")
        print(f"  - MSE: {mse:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R² Score: {r2:.4f}")
        print(f"  - Adjusted R² Score: {adjusted_r2:.4f}")
        
        # Draw prediction diagram
        """if plot:
            # Draw prediction vs. real value dot plot 
            plt.figure(figsize=(6, 6))
            plt.scatter(Y.numpy(), y_pred, alpha=0.5, color="red")
            plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], linestyle='--', color='black')
            plt.xlabel("Actual House Prices")
            plt.ylabel("Predicted House Prices")
            plt.title("Actual vs. Predicted House Prices")
            plt.show()"""

        return mse, rmse, r2, adjusted_r2

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    
    trained_model.to(device)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

"""
# code for using Ray Tune (LabTS does not have this package)
def train_regressor(config, x_train, y_train, x_val, y_val):

    model = Regressor(
        x=x_train, 
        nb_epoch=config["nb_epoch"], 
        hidden_size=config["hidden_size"], 
        num_hidden_layers=config["num_hidden_layers"],
        learning_rate=config["learning_rate"]
    )
    model.fit(x_train, y_train, batch_size=config["batch_size"])
    mse, _, r2, _ = model.score(x_val, y_val)
    tune.report({"loss":mse})
"""

def perform_hyperparameter_search(x_train, y_train): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    ########################  * Grid Search  *  ######################  

    """
    # Define the hyperparameter search space
    # code for using Ray Tune (LabTS does not have this package)
    param_grid = {
        "hidden_size": tune.grid_search([128, 256]), # 32, 64, 128, 256 
        "num_hidden_layers": tune.grid_search([3, 4, 5]), # 2, 3, 4, 5
        "learning_rate": tune.grid_search([0.001]),  
        "batch_size": tune.grid_search([64]),  
        "nb_epoch": tune.grid_search([2000]),  
        "dropout_rate": tune.grid_search([0.2])  
    }
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42) # Split into training & validation data set

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=2000,  # Max training iterations per trial
        grace_period=10,  # Allow trials to run at least 10 epochs before stopping
        reduction_factor=2  # Reduce poorly performing trials exponentially
    )

    reporter = CLIReporter(
        parameter_columns=list(param_grid.keys()),
        metric_columns=["loss", "training_iteration"]
    )

    result = tune.run(
        tune.with_parameters(train_regressor, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val),
        config=param_grid,
        num_samples=1, # 1 for grid search
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    best_config = best_trial.config

    print(f"\nBest Hyperparameters: {best_config}")
    print(f"Best Loss: {best_trial.last_result['loss']:.4f}")

    return best_config
    """

    param_grid = {
        "hidden_size": [256],  # Number of neurons in hidden layers # 32, 64, 128, 256 
        "num_hidden_layers": [4], # Number of hidden layers  # 2, 3, 4, 5
        "learning_rate": [0.001],  # Learning rate for optimizer
        "batch_size": [64],  # Mini-batch size
        "nb_epoch": [2000],  # Number of training epochs
        "dropout_rate": [0.2]  # Dropout rate
    }


    # Generate all possible hyperparameter combinations using itertools.product
    param_combinations = list(product(*param_grid.values()))

    best_score = np.inf  # Track the best MSE Score
    best_params = None  # Store the best hyperparameters
    best_model = None  # Store the best trained model

    total_combinations = len(param_combinations)
    print(f"Total hyperparameter combinations: {total_combinations}")

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42) # Split into training & validation data set

    # Iterate over all possible hyperparameter combinations
    for i, values in enumerate(param_combinations):
        hyperparams = dict(zip(param_grid.keys(), values))

        print(f"\n[{i+1}/{total_combinations}] Testing hyperparameters: {hyperparams}")

        # Initialize model with the sampled hyperparameters
        model = Regressor(
            x=x_train, 
            nb_epoch=hyperparams["nb_epoch"], 
            hidden_size=hyperparams["hidden_size"], 
            num_hidden_layers=hyperparams["num_hidden_layers"],
            learning_rate=hyperparams["learning_rate"],
            dropout_rate=hyperparams["dropout_rate"]

        )
        
        # Train the model
        model.fit(x_train, y_train, batch_size=hyperparams["batch_size"])

        # Evaluate the model
        mse, _, r2, _  = model.score(x_val, y_val) # Evaluate by validation data

        print(f"MSE Score: {mse:.4f}")

        # Save the best model if the MSE Score is lower
        if mse < best_score:
            best_score = mse
            best_params = hyperparams
            best_model = model  # Save the trained model

    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Best mse Score: {best_score:.4f}")

    # Save the best model
    save_regressor(best_model)
    return best_params


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    """
    Main function to run the model training, hyperparameter tuning, 
    evaluation, and model saving.
    """

    output_label = "median_house_value"

    # Load dataset
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # === Step 1: Split into Training and Testing Sets ===

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("\nPerforming hyperparameter tuning...\n")
    # === Step 2: Find Best Hyperparameters ===
    best_params = perform_hyperparameter_search(x_train, y_train) # We use the best hyperparameters found in the grid search    
    print("\nBest Hyperparameters Found:", best_params)

    # === Step 3: Train Final Model with Best Hyperparameters ===
    print("\nTraining final model with best hyperparameters...\n")
    regressor = Regressor(
        x_train, 
        nb_epoch=best_params["nb_epoch"], 
        hidden_size=best_params["hidden_size"], 
        num_hidden_layers=best_params["num_hidden_layers"], 
        learning_rate=best_params["learning_rate"]
    )

    regressor.fit(x_train, y_train, batch_size=best_params["batch_size"])

    # === Step 4: Save the Best Model ===
    save_regressor(regressor)

    # === Step 5: Load Model and Evaluate on Test Set ===
    loaded_regressor = load_regressor()

    print("\nEvaluating on training set...")
    train_mse, train_rmse, train_r2, train_adj_r2 = loaded_regressor.score(x_train, y_train, plot=True)
    print(f"\nTraining R² Score: {train_r2:.4f}, Adjusted R²: {train_adj_r2:.4f}")

    print("\nEvaluating on test set...")
    test_mse, test_rmse, test_r2, test_adj_r2 = loaded_regressor.score(x_test, y_test, plot=True)
    print(f"\nTest R² Score: {test_r2:.4f}, Adjusted R²: {test_adj_r2:.4f}")

if __name__ == "__main__":
    example_main()