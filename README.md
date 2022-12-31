There were two aspects to this project; to create a neural network mini library and create and train a neural network for regression.

### The first (mini-library) task required us to implement functions for the following: 
- Linear layers
    - Performs the XW+B transformation and relavent gradient calculations for backpropagation
- Activation functions
    - Implemented the sigmoid and ReLU functions, and performs relavent calculations for backpropagation
- Multi-layer networks 
    - Comprised of linear layers and activation functions as implemented previously and gives output of whole network when provided with input
    - Calculates gradient with respect to parameters and inputs of network
- Preprocessing of data
    - Performs scaling, formats and fills in gaps for the training data
- Trainer for the network
    - Handles data shuffling and performs mini-batched gradient descent on training data and computes loss on validation data.
    
### The second task was to create an train a neural network to predict house prices based on the given data:
This was achieved using Pytorch, Numpy, Scikit-learn and Pandas. The report for this has been included in this repository.
The following functions were created:
- Preprocessing
    - Pandas was used to import the dataset and various methods were used for preprocessing, such as one-hot encoding for textual values and filling in  gaps (various methods were tried as explained in the report)
- Model trainer
     - Performed a forward pass, calculates loss, performs a backwards pass to compute gradients and performs gradient descent using PyTorch. 
     - Other methods like batching, early stopping and shuffled data were used as outlined in the report.
- Model evaluation
    - A held-out dataset was used and the RMSE and R2 score were calculated since they were identified as useful parameters 
- Hyperparameter tuning
    - The learning rate, batch size and number of neurons in a hidden layer were tuned over 10 epochs using RMSE as the metric. 
    
The findings, analysis and justification of our choices can be found in our [report](Neural_Networks_Report.pdf).

The group worked on this project using PyCharm's code-with-me plugin, allowing us to simultaneously code on the same file using a member of the group's IDE as the host.





