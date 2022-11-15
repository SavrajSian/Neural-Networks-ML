import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing


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
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Model parameters
        self.model = torch.nn.Linear(self.input_size, self.output_size)
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

        # TODO: finish this function (very incomplete and probably wrong logic)

        # Fill missing values
        if self.missing_values is None:
            self.missing_values = {'longitude': 0, 'latitude': 0, 'housing_median_age': 0, 'total_rooms': 0,
                                   'total_bedrooms': 0,
                                   'population': 0, 'households': 0, 'median_income': 0, 'ocean_proximity': "NEARBY",
                                   'median_house_value': 0}
        print(x)
        x = x.fillna(value=self.missing_values)
        x_np = x.to_numpy()

        # 1-hot encoding textual value TODO: retain column order for later
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit_transform(x['ocean_proximity'])
        self.one_hot_cols = self.lb.classes_

        #x = pd.get_dummies(data=x, columns=['ocean_proximity'])


        # Normalise numerical values
        # Could be replaced by Pandas/PyTorch function?
        self.a = 0
        self.b = 1

        if training:
            self._min = np.amin(x)
            self._max = np.amax(x)

        factor = (self.b - self.a) / (self._max - self._min)
        inverse_factor = (self._max - self._min) / (self.b - self.a)

        self.normalize = lambda y: self.a + (y - self._min) * factor
        self.retrieve = lambda y: self._min + (y - self.a) * inverse_factor

        return x, (y if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
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
        # TODO: any additional stuff such as batch learning
        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget

        for i in range(self.nb_epoch):
            # Forward pass
            predictions = self.model(X)

            # Calculate loss
            loss = self.loss_fn(predictions, Y)

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

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        pass

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
        return 0  # Replace this code with your own

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
    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train)
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

    # print("GUYS PART 2 IS VERY COMPLICATED")
    # print("SO WE SHOULD NOW HAVE 3 PEOPLE WORKING ON PART 2, AND 1 PERSON DEBUGGING PART 1.")
    # print("OTHERWISE WE WON'T HAVE ENOUGH TIME.")
