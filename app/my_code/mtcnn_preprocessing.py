import os
from PIL import Image
from facenet_pytorch import MTCNN

repo_root = os.path.abspath(os.path.join(os.getcwd(),".."))
data_dir = os.path.join(repo_root, "data")

classes = ["happy", "angry", "sad"]

# output images of detected faces of size 160x160
mtcnn = MTCNN(image_size=160, select_largest=True, margin=20, min_face_size=10)
filename = None

for class_ in classes:
	class_directory = os.path.join(data_dir, class_)
	for filename in os.listdir(class_directory):
		if filename[-4:] != ".jpg":
			continue
		image_dir = os.path.join(class_directory, filename)
		my_image = Image.open(image_dir)
		save_path = os.path.join(data_dir, f"{class_}_processed", filename)
		# mtcnn does not return anything, since I only use it to save the cropped image
		# of each detected face in a new directory
		mtcnn(my_image, save_path=save_path)
