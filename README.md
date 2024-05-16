## Requirements:
* python 3.10+
* pytorch and torchvision
* numpy
* pandas
* matplotlib
* beautiful soup (bs4)
* scipy
* sklearn
* PIL

A requirements file is included but if that does not work please install these.
# Data
The data used to train the models can be found at https://diuf.unifr.ch/main/hisdoc/sites/diuf.unifr.ch.main.hisdoc/files/uploads/diva-hisdb/hisdoc/all.zip (NOTE THIS LINK WILL START THE DOWNLOAD)

Just extract the folder all and put it in the working directory

Keep all python files and notebooks in the same directory

To run 'test.ipynb' and certain functions in 'visualisations.ipynb' you might have to change the path to redirect them to the folder where you store your model weights that are produced by 'train.ipynb'.

the following variables might need to be changed to match a your path:
* model_weights_file_path = "D:\Hämtade Filer\models-20240320T180531Z-001\models"
* model_weights_file_path = "modelsDeeplab"


# File description
* 'test.ipynb' contains the inference and testing of said inference
* 'training.ipynb' was used to generate model weights
* 'visualisations.ipynb' contains various visualisation methods and cells used during the course of the project
* 'utils.py' contains functions which are used in one or more notebooks.

# To do 
* Implement L-U-Net AttentionU-Net in pytorch
* Possibly revise U-Net
* Benchmark the models
* Decide downscaling size
* Data augmentation
