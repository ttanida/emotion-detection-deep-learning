import streamlit as st

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate

from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN

from my_code.model import EmotionDetectionModel

model = EmotionDetectionModel.load_from_checkpoint('model_weights_of_best_model/epoch=25-val_loss=0.36.ckpt')
model.eval()


def make_inference(img):
	"""  Makes inference on single image.
	Returns image with predictions drawn on and class probs as a torch tensor

	angry: class 0
	happy: class 1
	sad: class 2

	Params:
	-------
	img (PIL)

	Returns:
	-------
	img_draw (PIL)
	probs (torch tensor)"""

	with torch.no_grad():
		# use mtcnn to extract the face in the image
		mtcnn = MTCNN(image_size=160, select_largest=False, margin=20, min_face_size=10, post_process=True, thresholds=[0.8, 0.9, 0.9])
		face_img = mtcnn(img)

		if face_img is None:  # if no face was detected
			# use MTCNN with lower threshold for detecting faces
			mtcnn = MTCNN(image_size=160, select_largest=False, margin=20, min_face_size=10, post_process=True,
						  thresholds=[0.4, 0.5, 0.5])
			face_img = mtcnn(img)
			if face_img is None:  # if still no face was detected
				return None, None
			else:
				# weirdly, the outputted faces are rotated clockwise for 90° for the lower threshold,
				# so we have to rotate it counter-clockwise to get the correct face image
				face_img = rotate(face_img, angle=-90)

		# get the scores and corresponding class probs
		scores = model(face_img).squeeze()
		probs = F.softmax(scores, dim=0)

		# draw a red box around the detected face with the class probs underneath it
		img_draw = draw_box_with_probs(img, probs, mtcnn)

	return img_draw, probs

def draw_box_with_probs(img, probs, mtcnn):
	"""Returns image with red box around detected face and the class probs displayed underneath it
	Params:
	-------
	img (PIL)
	probs (torch tensor)

	Returns:
	-------
	img_draw (PIL), image with red box and class probs on it"""

	# draw a red box around detected face
	box, _ = mtcnn.detect(img)
	img_draw = img.copy()
	draw = ImageDraw.Draw(img_draw)
	draw.rectangle(box[0], width=5, outline=(255, 0, 0))

	# x and y coordinates specify the lower left corner of the box
	# this is where the class probs will be displayed
	x_coordinate = box[0][0]
	y_coordinate = box[0][3]

	# since images have variable resolutions, we need to adjust the font size accordingly
	fontsize = 1  # starting font size
	img_fraction = 0.4 # portion of image width the text width is going to be

	txt = "probability class \"angry\": 80%"  # sample txt to see how big the font has to be
	font_path = "font/Arialnb.ttf"

	font = ImageFont.truetype(font_path, fontsize)
	while font.getsize(txt)[0] < img_fraction * img_draw.size[0]:
		# iterate until the text size is just larger than the criteria
		fontsize += 1
		font = ImageFont.truetype(font_path, fontsize)

	fontsize -= 1  # decrease the font size by one just in case
	font = ImageFont.truetype(font_path, fontsize)  # final font specifications

	# index of predicted class
	pred_class_index = probs.argmax().item()

	# write the text with the class probs
	classes = ['angry', 'happy', 'sad']
	text = ""
	for i, class_ in enumerate(classes):
		text += f"probability class \"{class_}\": {probs[i]:.0%}\n"
	text += f"-> Predicted class: \"{classes[pred_class_index]}\""

	draw.multiline_text((x_coordinate, y_coordinate), text=text, font=font, fill=(255, 0, 0, 0), align="left", spacing=3)

	return img_draw


def main():
	st.title("Emotion Detection Model")
	st.write("## How does it work?")
	st.write(
		"Add an image of a person and a deep learning model will classify it like in the example below:")
	st.image(Image.open("sample images/classified_image.jpg"),
			 caption="Example of model being run on an image.",
			 use_column_width=True)
	st.write("## Upload your own image")
	st.write("note: if the upload fails, please click the 'x' on the right and try uploading again")
	st.write("Sometimes uploading takes a couple of tries.")
	uploaded_image = st.file_uploader("Choose a png or jpg image",
									  type=["jpg", "png", "jpeg"])

	if uploaded_image is not None:
		image = Image.open(uploaded_image)

		# Make sure image is RGB
		image = image.convert("RGB")

		if st.button("Make a prediction"):
			with st.spinner("Doing the math..."):
				img_draw, probs = make_inference(image)
				if img_draw is None:
					st.write("Face could not be detected... :(")
				else:
					st.write("Outputting prediction...")
					classes = ['angry', 'happy', 'sad']
					pred_class_index = probs.argmax().item()
					probs = probs.tolist()
					st.image(img_draw, caption=f"angry: {probs[0]:.0%}, happy: {probs[1]:.0%}, sad: {probs[2]:.0%}", use_column_width=True)
					st.write(f"angry: {probs[0]:.0%}, happy: {probs[1]:.0%}, sad: {probs[2]:.0%}")
					st.write(f"predicted class: \"{classes[pred_class_index]}\"")

	st.write("## How is this made?")
	st.write("The machine learning happens with a fine-tuned [Inception Resnet V1](https://github.com/timesler/facenet-pytorch) model (PyTorch), \
    this front end is built with [Streamlit](https://www.streamlit.io/) \
    and it's all hosted on [Google's App Engine](https://cloud.google.com/appengine/).")
	st.write(
		"See the [code on GitHub](https://github.com/ttanida/emotion-detection-deep-learning)")


if __name__ == "__main__":
	main()
