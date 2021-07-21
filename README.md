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

(1) my model always receives images of the same resolution (160x160), so I can design the model architecture accordingly <br />
(2) my model can use its full "power" on predicting emotions for the given face, and doesn't have to first find a face in an image

<img src="https://github.com/ttanida/emotion-detection-deep-learning/blob/main/images_for_README/model_overview.png">
