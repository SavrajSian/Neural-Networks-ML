import math
import torch
from torch import nn
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler


class Regressor:

    def __init__(self, x, lr=0.014, nb_epoch=250, neurons_per_hidden_layer=[64], batch_size=16):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.missing_values = None
        self.one_hot_cols = None
        self.r2_score = 0

        X, _ = self._preprocessor(x, training=True)

        self.input_size = X.shape[1]  # Number of features
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        # Neural network variables
        self.loss_fn = nn.MSELoss()

        npl = [*neurons_per_hidden_layer, self.output_size]

        layers = [nn.Linear(self.input_size, npl[0])]
        for i in range(len(npl) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(npl[i], npl[i + 1]))

        self.model = nn.Sequential(*layers)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    # Converts numpy array to torch tensor
    def numpy_to_tensor(self, x):
        return torch.from_numpy(x).float()

    def _preprocessor(self, x, y=None, training=False):
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
        # Return preprocessed x and y, return None for y if it was None
        x_pre = x

        # Fill missing values
        if self.missing_values is None:
            self.missing_values = {'ocean_proximity': x["ocean_proximity"].mode().to_string()}
            for col in set(x.columns.values).difference({'ocean_proximity'}):
                self.missing_values[col] = x[col].mean()

        x_pre = x_pre.fillna(value=self.missing_values)

        # 1-hot encoding textual value
        if training:
            self.lb = preprocessing.LabelBinarizer()
            self.lb.fit(x_pre['ocean_proximity'])
            self.one_hot_cols = self.lb.classes_

        enc = self.lb.transform(x_pre['ocean_proximity'])
        x_pre = x_pre.drop('ocean_proximity', axis=1)
        for i, col in enumerate(self.one_hot_cols):
            x_pre[str(col)] = enc[:, i]

        if training:
            self.scaler_x = StandardScaler()
            self.scaler_x.fit(x_pre)
            if y is not None:
                self.scaler_y = StandardScaler()
                self.scaler_y.fit(y)

        y_pre = y if y is None else self.scaler_y.transform(y)
        return self.scaler_x.transform(x_pre), y_pre
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Perform forward pass though the model given the input.
        • Compute the loss based on this forward pass.
        • Perform backwards pass to compute gradients of loss with respect to parameters of the model.
        • Perform one step of gradient descent on the model parameters.
        • You are free to implement any additional steps to improve learning (batch-learning, shuffling...)

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
        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        len_X = X.shape[0]
        self.model.train(True)

        for i in range(self.nb_epoch):
            for j in range(0, len_X, self.batch_size):
                x_tensor = self.numpy_to_tensor(X[j: min(len_X, j + self.batch_size)])
                y_tensor = self.numpy_to_tensor(Y[j: min(len_X, j + self.batch_size)])

                # Forward pass
                predictions = self.model(x_tensor)

                # Calculate loss
                loss = self.loss_fn(predictions, y_tensor)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient descent
                self.optimizer.step()

        return self
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
        if isinstance(x, pd.DataFrame):
            x_pre, _ = self._preprocessor(x, training=False)
            X = self.numpy_to_tensor(x_pre)
        elif isinstance(x, np.ndarray):
            X = self.numpy_to_tensor(x)
        elif isinstance(x, torch.Tensor):
            X = x
        else:
            print(f"Invalid input type: {type(x)}")
            return None

        self.model.eval()
        prediction_scaled = self.model(X).detach().numpy()
        prediction = self.scaler_y.inverse_transform(prediction_scaled)
        return prediction
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
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
        self.model.eval()

        X, _ = self._preprocessor(x, y=y, training=False)  # Do not forget
        X_tensor = self.numpy_to_tensor(X)

        predicted_labels = self.predict(X_tensor)
        true_labels = y.to_numpy()

        if np.isnan(predicted_labels).any():
            return float('inf')

        self.r2_score = metrics.r2_score(true_labels, predicted_labels)
        return metrics.mean_squared_error(true_labels, predicted_labels, squared=False)
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
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch():
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
    # Params being tuned: learning_rate, batch_size, number of neurons in hidden layer
    params = {"learning_rate": 0.0, "batch_size": 0, "hidden_neurons": 0}

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")
    data_shuffled = data.sample(frac=1)

    x = data_shuffled.loc[:, data_shuffled.columns != output_label]
    y = data_shuffled.loc[:, [output_label]]

    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    models = []

    best_error = float('inf')
    for lr_scaled in range(1, 51):  # 0.002 -> 0.004 -> ... -> 0.1 (50 values)
        for batch_power in range(3, 8):  # 8 -> 16 -> 32 -> 64 -> 128 (5 values)
            for hn_scaled in range(1, 11):  # 8 -> 16 -> ... -> 80 (10 values)
                learning_rate = float(lr_scaled) / 500
                batch_size = 2 ** batch_power
                hidden_neurons = hn_scaled * 8
                reg = Regressor(x_train, lr=learning_rate, nb_epoch=10,
                                neurons_per_hidden_layer=[hidden_neurons], batch_size=batch_size)
                reg.fit(x_train, y_train)
                error = reg.score(x_test, y_test)
                if math.isinf(error):
                    continue
                models.append([learning_rate, batch_size, hidden_neurons, error, reg.r2_score])
                if error < best_error:
                    print(f"\nNew best params: {learning_rate}, {batch_size}, {hidden_neurons}")
                    print(f"New best RMSE: {error}")
                    best_error = error
                    params["learning_rate"] = learning_rate
                    params["batch_size"] = batch_size
                    params["hidden_neurons"] = hidden_neurons

    print(f"\nBest learning rate: {params['learning_rate']}")
    print(f"Best batch size: {params['batch_size']}")
    print(f"Best hidden neurons: {params['hidden_neurons']}")
    print(f"Best error: {best_error}")

    df = pd.DataFrame(models, columns=['Learning Rate', 'Batch Size', '# Hidden Neurons', 'RMSE', 'R2'])
    df.to_csv('models.csv', index=False)
    print("CSV successfully created.")

    return params  # Return the chosen hyper-parameters
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    print("Creating regressor")
    regressor = Regressor(x_train)
    print("Fitting data")
    regressor.fit(x_train, y_train)
    print("Save to file")
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


def hyperparameter_main(params):
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    print("Creating regressor")
    regressor = Regressor(x_train,
                          lr=params["learning_rate"],
                          neurons_per_hidden_layer=[params["hidden_neurons"]],
                          batch_size=params["batch_size"]
                          )
    print("Fitting data")
    regressor.fit(x_train, y_train)
    print("Save to file")
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    # parameters = RegressorHyperParameterSearch()
    # hyperparameter_main(parameters)
    example_main()
