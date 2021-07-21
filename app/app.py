import streamlit as st

import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, extract_face

from my_code.model import EmotionDetectionModel

model = EmotionDetectionModel.load_from_checkpoint('models/epoch=25-val_loss=0.36.ckpt')
model.eval()


def make_inference(img):
	"""  Makes inference on single image.
	Returns image with predictions drawn on and class probs as a list

	angry: class 0
	happy: class 1
	sad: class 2

	Params:
	-------
	img (PIL)"""

	with torch.no_grad():
		mtcnn = MTCNN(image_size=160, select_largest=False, margin=20, min_face_size=10, post_process=True, thresholds=[0.8, 0.9, 0.9])
        face_in_image = mtcnn(image)
        
		if face_img is None: # if no face was detected
			probs = [1/3, 1/3, 1/3]
			return img, probs
		else:
			scores = model(face_img).squeeze()
			probs = F.softmax(scores, dim=0)

			# draw a red box around detected face
			box, _ = mtcnn.detect(img)
			img_draw = img.copy()
			draw = ImageDraw.Draw(img_draw)
			draw.rectangle(box[0], width=5, outline=(255, 0, 0))

			classes = ['angry', 'happy', 'sad']
			x_coordinate = box[0][0]
			y_coordinate = box[0][3]

			txt = "probability class \"angry\": 80%"
			fontsize = 1  # starting font size

			# portion of image width you want text width to be
			img_fraction = 0.5

			font_path = "font/Arialnb.ttf"
			font = ImageFont.truetype(font_path, fontsize)
			while font.getsize(txt)[0] < img_fraction * img_draw.size[0]:
				# iterate until the text size is just larger than the criteria
				fontsize += 1
				font = ImageFont.truetype(font_path, fontsize)

			fontsize -= 1
			font = ImageFont.truetype(font_path, fontsize)
			text = ""
			for i, class_ in enumerate(classes):
				text += f"probability class \"{class_}\": {probs[i]:.0%}\n"

			pred_class_index = probs.argmax().item()
			text += f"-> Predicted class: \"{classes[pred_class_index]}\""

			draw.multiline_text((x_coordinate, y_coordinate), text=text, font=font, fill=(255,0,0,0), align="left", spacing=3)
			probs = probs.tolist()

	return img_draw, probs

def main():
	st.title("Emotion Detection Model")
	st.write("## How does it work?")
	st.write(
		"Add an image of a person and a deep learning model will classify it like in the example below:")
	st.image(Image.open("sample images/classified_image.jpg"),
			 caption="Example of model being run on an image.",
			 use_column_width=True)
	st.write("## Upload your own image")
	uploaded_image = st.file_uploader("Choose a png or jpg image",
									  type=["jpg", "png", "jpeg"])

	if uploaded_image is not None:
		image = Image.open(uploaded_image)

		# Make sure image is RGB
		image = image.convert("RGB")

		if st.button("Make a prediction"):
			with st.spinner("Doing the math..."):
				img_draw, probs = make_inference(image)
				st.image(img_draw, caption=f"angry: {probs[0]:.0%}, happy: {probs[1]:.0%}, sad: {probs[2]:.0%}", use_column_width=True)
				st.write(f"angry: {probs[0]:.0%}, happy: {probs[1]:.0%}, sad: {probs[2]:.0%}")

	st.write("## How is this made?")
	st.write("The machine learning happens with a fine-tuned [Inception Resnet V1](https://github.com/timesler/facenet-pytorch) model (PyTorch), \
    this front end is built with [Streamlit](https://www.streamlit.io/) \
    and it's all hosted on [Google's App Engine](https://cloud.google.com/appengine/).")
	st.write(
		"See the [code on GitHub](https://github.com/ttanida/emotion-detection-deep-learning)")


if __name__ == "__main__":
	main()
