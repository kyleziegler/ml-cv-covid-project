# Tensorflow for Computer Vision Project
Leveraging TensorFlow to create bounding boxes and classes on CT scans of the lungs.

## Data information
- [Kaggle](https://www.kaggle.com/datasets/hgunraj/covidxct), 60GB, 200k images with bounding boxes and classes. 

## To get started
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/hgunraj/covidxct)
2. Use the data_prep.py script to create TF records
3. Use the model.ipynb file to create and fit the model


TODO
- Allow the user to start from a previous checkpoint, load save model file.
- Convert all notebooks to python scripts, create classes, and allow the entire process to be kicked off from scipts.
- Define a Dockerfile
- Add functionality to let the user choose their own metrics and loss function
- Show the results from my training job, and provide an analysis
- Test out batch normalization vs instance based normalization
- Write functional tests for this project
