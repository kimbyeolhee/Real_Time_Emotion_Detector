# Data source
Data was downloaded from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013

## About Dataset
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The task was originally to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

**But i changed dataset to categorize (0=Angry, 1=Happy, Neutral=3, Sad=4) to remove other label's image files.**



## Data folder structure
```
├─README.md
├─train
│ ├─angry
│ ├─happy
│ ├─neutral
│ └─sad
├─test
│ ├─angry
│ ├─happy
│ ├─neutral
│ └─sad
├─labels.txt
├─train.csv
└─test.csv
```
