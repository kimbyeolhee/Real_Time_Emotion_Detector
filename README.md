<h1 >Real Time Emotion Detector</h1>
Using cam to capture face emotion into 4 category (Happy, Sad, Neutral, Angry)

## Environment setup

We have experimented the implementation on the following enviornment.

```
# create virtual python enviroment
python -m venv venv_vit
source venv_vit/Scripts/activate

# install required libraries
pip install -r requirements.txt
```

## Prepare dataset

Datasets we used are as follows:
| Dataset | Download | Comment |
|-------|:--------------------:|---------------|
| FER-2013 | [Link](https://www.kaggle.com/datasets/msambare/fer2013) | Change categories 7 to 4 |

For more details, please refer to [data description](/data/README.md).

## Train model

```
python train.py -c base_config
```

## Detect emotion

Download the [haarcascade_frontface_default.xml file](https://github.com/opencv/opencv/tree/master/data/haarcascades) and place it in the haarcascade_files folder.

You can use the trained model by specifying the base_config from Hugging Face ([StoneSeller/emotion-classifier-vit](https://huggingface.co/StoneSeller/emotion-classifier-vit)).

```
python inference.py -c inference_config
```
