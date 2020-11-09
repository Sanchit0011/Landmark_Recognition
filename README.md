# Landmark_Recognition

The goal of this challenge is to build a model that recognizes the correct landmark (if any) in a dataset of challenging test images. 
The data set used for this challenge is the Google-Landmarks Dataset which contains URLs of images publicly available online.

## Steps to Reproduce
1. Download *train.csv* from https://www.kaggle.com/google/google-landmarks-dataset
2. Run *data_sampling.py* to retrieve 2% images from the top 100 most frequent classes
3. Create *train_images*, *valid_images* and *test_images* directories to store train, validation and test images
4. Run *directory_structure.py* to download images and split them between the directories
5. Create *saved_models* directory to save Convolutional Neural Network model weights
6. Run *Landmark_Recognition.ipynb* to train and evaluate base Convolutional Neural Network for landmark recognition
