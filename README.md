## The code contains a custom resnet model used in the training of CIFAR dataset for achieving 90% accuracy

### Here is the [REPO](https://github.com/deepanshudashora/ERAV1/tree/master/session10) for that 

#### File Structure

* [custom_models](https://github.com/deepanshudashora/custom_models) -> A Repository contains files for training

  * [models](https://github.com/deepanshudashora/custom_models/tree/main/models) -> For importing model architecture, for this particular problem statement we will be importing resnet.py file and using resnet18 model
  * [train.py](https://github.com/deepanshudashora/custom_models/blob/main/train.py) -> Contains training loop
  * [test.py](https://github.com/deepanshudashora/custom_models/blob/main/test.py) -> Contains code for running model on the test set
  * [utils.py](https://github.com/deepanshudashora/custom_models/blob/main/utils.py) -> Contains supportive functions
  * [main.py](https://github.com/deepanshudashora/custom_models/blob/main/main.py) -> Contains code for fitting model in training and testing loops
  * [gradcam_utils.py](https://github.com/deepanshudashora/custom_models/blob/main/gradcam_utils.py) -> Contains code for running gradcam on model 
