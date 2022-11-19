import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class Regressor():
    missing_values = None
    one_hot_cols = None

    def __init__(self, x, nb_epoch=1000):
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

        # Replace this code with your own
        # TODO: Complete function
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]  # Number of features
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Model parameters
        self.model = torch.nn.Linear(self.input_size, self.output_size).double()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03)
        return

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

        # Fill missing values TODO: use kNN/test median,mode,other metrics
        m = x['total_bedrooms'].mean()
        if self.missing_values is None:
            self.missing_values = {'longitude': x["longitude"].mean(), 'latitude': x["latitude"].mean(),
                                   'housing_median_age': x["housing_median_age"].mean(),
                                   'total_rooms': x["total_rooms"].mean(),
                                   'total_bedrooms': x["total_bedrooms"].mean(),
                                   'population': x["population"].mean(), 'households': x["households"].mean(),
                                   'median_income': x["median_income"].mean(),
                                   'ocean_proximity': x["ocean_proximity"].mode().to_string()}
        x = x.fillna(value=self.missing_values)

        # 1-hot encoding textual value TODO: retain column order for later
        if training:
            self.lb = preprocessing.LabelBinarizer()
            self.lb.fit_transform(x['ocean_proximity'])
            self.one_hot_cols = self.lb.classes_

        enc = self.lb.fit_transform(x['ocean_proximity'])
        x = x.drop('ocean_proximity', axis=1)
        for i, col in enumerate(self.one_hot_cols):
            x[str(col)] = enc[:, i]

        if training:
            self.scaler = MinMaxScaler() #TODO: try other scalers eg 0 mean unit variance
            self.scaler.fit(x)

        x = self.scaler.transform(x)
        pd.set_option('display.max_columns', None)
        return x, (y if isinstance(y, pd.DataFrame) else None)

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

        for i in range(self.nb_epoch):
            # Forward pass
            predictions = self.model(torch.from_numpy(X))
            # Calculate loss
            loss = self.loss_fn(predictions.double(), torch.Tensor(Y.values).double())

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

        # X, _ = self._preprocessor(x, training=False)  # Do not forget
        return self.model(x)

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

        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        predicted_labels = []
        for X_point in X:
            predicted_labels.append(self.predict(torch.from_numpy(X_point)).item())
        print(y.values.tolist()[:10])
        print(predicted_labels[:10])
        return metrics.r2_score(y.values.tolist(), predicted_labels)  # Replace this code with your own

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

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
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
    regressor = Regressor(x_train, nb_epoch=10)
    print("Fitting data")
    regressor.fit(x_train, y_train)
    print("Save to file")
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

    output_label = "median_house_value"

    data = pd.read_csv("housing.csv")

    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    regressor = Regressor(x_train, nb_epoch=10)
