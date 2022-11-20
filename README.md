This is an implementation of the MNIST experiment from the paper "Group Equivariant Convolutional Networks" by Taco S. Cohen, Max Welling in Pytorch Lightning. The p4 group convolutional layers are implemented for the p4cnn (as described in the paper) as well as the baseline model z2cnn. 

### Dataset
The MNIST dataset is used, which is automatically downloaded. The images of the dataset are randomly rotated. 

### Packages

You can run the following command to install all the packages listed in the requirements.txt:

`pip3 install -r requirements.txt`

## Run

Run the following command 
 
In order to evaluate the model, run

`python train_mnist.py --model [MODEL_NAME]`

where `[MODEL_NAME]` in ["z2cnn", "p4cnn"] .

Tensorboard reports can be accessed via 

`tensorboard --logdir lightning_logs/`

If we rotate the MNIST pictures of the validation set,but train on a non-rotated training set, the p4cnn outperforms z2cnn, through the inductive bias of the network

Achieved validation accuracy (rotated validation set) at set parameters: 
- z2cnn: 42%
- p4cnn: 84% 


## References

T.S. Cohen, M. Welling, Group Equivariant Convolutional Networks. Proceedings of the International Conference on Machine Learning (ICML), 2016.

Other work consulted: 

https://www.youtube.com/watch?v=z2OEyUgSH2c&list=PL8FnQMH2k7jzPrxqdYufoiYVHim8PyZWd&ab_channel=ErikBekkers
https://www.youtube.com/watch?v=TOfg-JlLILA&ab_channel=PreserveKnowledge
https://github.com/tscohen/GrouPy
https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.ipynb#scrollTo=3XblvSZl_ie9
https://github.com/claudio-unipv/groupcnn/blob/main/mnist.py
https://github.com/tueimage/SE2CNN
