# emotion-detection-deep-learning
A pytorch model that classifies images of people to 3 distinct emotions

<img src="https://github.com/ttanida/emotion-detection-deep-learning/blob/main/images_for_README/sample_img.jpeg" width="650" height="700" />

## Try it for yourself

You can try the [live and deployed app which uses the model](https://emotion-detection-320415.ew.r.appspot.com/) (note: this may have been taken down by the time you check it due to costs).

Just upload an image and press "Make a prediction".

## How does it work?

First, an already trained [Multi-task Cascaded Convolutional Networks](https://github.com/timesler/facenet-pytorch/) (MTCNN) is used to detect and extract the face in the image.
Afterwards, my model uses the extracted face to output the probabilities for each class.

This has 2 benefits: 

(1) my model always receives images of the same resolution (160x160), so the model architecture is designed accordingly <br />
(2) my model can use its "full power" on predicting emotions for the given face, and doesn't have to first find a face in an image

<img src="https://github.com/ttanida/emotion-detection-deep-learning/blob/main/images_for_README/model_overview.png">

## How was the model trained?

I used transfer learning by using a [Inception Resnet (V1) model](https://github.com/timesler/facenet-pytorch/) pretrained on VGGFace2 as a feature extractor (by freezing all but the last 3 layers).

I trained a newly initizalized 1x1 conv layer (to reduce the channel dim of the feature maps) and 3 newly initizalized linear layers (to gradually decrease the output dim to 3 classes) with training data webscraped from Google Images. The model file can be found [here](https://github.com/ttanida/emotion-detection-deep-learning/blob/main/app/my_code/model.py).

The dataset consisted of 1507 images, roughly 500 images per class. It can found in [this google drive folder](https://drive.google.com/drive/folders/1h94EmiPXh3lMVtnwVH45kLaCh64Glp6t?usp=sharing).

I used various transformations (RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur) on the train set to augment the data.

The whole training procedure is described in the jupyter notebook [emotion_detection.ipynb](https://github.com/ttanida/emotion-detection-deep-learning/blob/main/emotion_detection.ipynb)

The final model achieved a test set accuracy of 88%.
