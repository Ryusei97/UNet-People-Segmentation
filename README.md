# UNet-People-Segmentation

This is an image segmentation project for people using UNets. The goal with this project is to create a model that can segment people in an image and use it to automatically remove backgrounds. `project_report.pdf` contains a summary of this project. The UNet architecture used in the project can be found in `utils.py` and the model diagram is shown below. 

## Files

- `project_report.pdf`: summary of the project
- `utils.py`: UNet architecture using tensorflow
- `dataset_exploration.ipynb`: Creates masks and reorganizes the downloaded dataset
- `train.ipynb`: Training notebook for the UNet
- `main.py`: Applies the trained model to webcam footage


![Model Architecture](https://github.com/Ryusei97/UNet-People-Segmentation/blob/main/plots/model_shape.png)

## Results

The training curves and sample predictions on the validation set are shown below. 

![Trainig Curves](https://github.com/Ryusei97/UNet-People-Segmentation/blob/main/plots/training_curves.png)

![Segmentation Results](https://github.com/Ryusei97/UNet-People-Segmentation/blob/main/plots/segmentation%20results.png)



## Datasets
Here are the datasets used for this project with some sample images/masks.

- [People - Segmentation](https://www.kaggle.com/datasets/quantigoai/people-segmentation)

![People - Segmentation](https://github.com/Ryusei97/UNet-People-Segmentation/blob/main/plots/dataset1.png)

- [Persons](https://ecosystem.supervisely.com/projects/persons)
  
![Persons](https://github.com/Ryusei97/UNet-People-Segmentation/blob/main/plots/dataset2.png)
