import torch
from torch import nn
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy.random import default_rng


class Regressor:


    def __init__(self, x, lr=0.1, nb_epoch=10, neurons_per_hidden_layer=[64], batch_size=64):
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
        # TODO: Complete function
        self.missing_values = None
        self.one_hot_cols = None
        self.batch_size = batch_size
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]  # Number of features
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Neural network variables
        self.loss_fn = nn.MSELoss(reduction='sum')

        npl = [*neurons_per_hidden_layer, self.output_size]

        layers = [nn.Linear(self.input_size, npl[0])]
        for i in range(len(npl) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(npl[i], npl[i + 1]))

        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        print(self.model)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

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

        # Fill missing values TODO: use kNN/test median,mode,other metrics
        if self.missing_values is None:
            self.missing_values = {'longitude': x["longitude"].mean(), 'latitude': x["latitude"].mean(),
                                   'housing_median_age': x["housing_median_age"].mean(),
                                   'total_rooms': x["total_rooms"].mean(),
                                   'total_bedrooms': x["total_bedrooms"].mean(),
                                   'population': x["population"].mean(), 'households': x["households"].mean(),
                                   'median_income': x["median_income"].mean(),
                                   'ocean_proximity': x["ocean_proximity"].mode().to_string()}
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
            self.scaler = StandardScaler()  # TODO: try other scalers eg 0 mean unit variance
            self.scaler.fit(x_pre)
            if y is not None:
                self.scy = StandardScaler()
                self.scy.fit(y)

        x_out = self.scaler.transform(x_pre)
        if y is not None:
            y = self.scy.transform(y)

        return x_out, y

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
        # TODO: any additional stuff such as batch learning
        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        print(Y)
        self.model.train(True)

        for i in range(self.nb_epoch):

            for i in range(0, X.shape[0], self.batch_size):
                y_tensor = torch.from_numpy(Y[i:i+self.batch_size]).float()

                # Forward pass
                predictions = self.model(torch.from_numpy(X[i:i+self.batch_size]).float())

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
            X, _ = self._preprocessor(x, training=False)
        elif isinstance(x, np.ndarray):
            X = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            X = x
        else:
            print(f"Invalid input type: {type(x)}")
            return None

        self.model.eval()
        y = self.model(X)
        y = self.scy.inverse_transform(y.detach().numpy())
        return y
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

        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        X_tensor = torch.from_numpy(X).float()

        predicted_labels = self.predict(X_tensor) #list(map(lambda t: self.predict(t).item(), X_tensor))
        true_labels_scaled = Y.flatten().tolist()
        # true_labels = y.flatten().to_list()


        # print(f"\nPredicted: {predicted_labels[:10]}")
        # print(f"True: {y.values.tolist()[:10]}")
        # print(f"\nDifference: {(np.array(y[:10]) - np.array(predicted_labels[:10])).tolist()}")

        return metrics.mean_squared_error(y, predicted_labels, squared=False)
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

    params = {"learning_rate": 0.01}
    data = pd.read_csv("housing.csv")
    output_label = "median_house_value"

    x = data.loc[:, data.columns != output_label].sample(frac=1)
    y = data.loc[:, [output_label]].sample(frac=1)

    split_idx = int(0.8 * len(x))
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    best_score = float('inf')
    for learning_rate in range(10, 5000, 100):
        for batch_size in range(8, 128, 8):
            for layer in range(6, 124, 6):
                lr = learning_rate / 10000
                print(f"Params set to: {lr}, {batch_size}, {layer}")
                reg = Regressor(x_train, lr=lr, batch_size=64, nb_epoch=10, neurons_per_hidden_layer=[12, 24])
                reg.fit(x_train, y_train)
                try:
                    score = reg.score(x_test, y_test)
                    if score < best_score:
                        print(f"Params resulted in new best score: {score}")
                        best_score = score
                        params["learning_rate"] = lr
                except:
                    print("failed")

    print(f"Best learning rate: {params['learning_rate']}")
    print(f"Best error: {best_score}")
    return params  # Return the chosen hyper-parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    # RegressorHyperParameterSearch()
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")
    data.isna().sum()

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    print("Creating regressor")
    regressor = Regressor(x_train, lr=0.01, nb_epoch=100, neurons_per_hidden_layer=[10, 10], batch_size=100)
    print("Fitting data")
    regressor.fit(x_train, y_train)
    print("Save to file")
    # save_regressor(regressor) TODO uncomment

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
