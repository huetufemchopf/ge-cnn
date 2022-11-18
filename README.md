This is an implementation of the MNIST experiment from the paper "Group Equivariant Convolutional Networks" by Taco S. Cohen, Max Welling. The p4 group convolutional layers are implemented for the p4cnn (as described in the paper) as well as the baseline model z2cnn. 

### Dataset
The MNIST dataset is used, which is automatically downloaded. The images of the dataset are randomly rotated. 

### Packages

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Python 3.6.13

## Run

Run the following command 
 
In order to evaluate the model, run

`python train_mnist.py --model [MODEL_NAME]`

where `[MODEL_NAME]` in [z2cnn, p4cnn] .

Achieved validation accuracy at set parameters: 
- z2cnn: 92.8 % (comparison paper: 100 - (0.1038 ± 0.27) %)
- p4cnn: (comparison paper: )

Fun fact: If we do not rotate the MNIST pictures of the training set, but keep the rotations on the test set, 
the z2cnn achieves an accuracy of 42%, whereas the p4cnn, through the inductive bias of the network, achieves still 83%.

