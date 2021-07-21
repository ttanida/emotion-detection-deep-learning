from facenet_pytorch import MTCNN

class DetectFaceAndTransformToTensor(object):
    """Convert a ``PIL Image`` of a face to a tensor of shape (3 x 224 x 224).
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, image):
        """
        Args:
            image (PIL Image): Image to be converted to tensor of face.
        Returns:
            Tensor: Converted image of shape (3 x 224 x 224) of the detected face.
        """

        mtcnn = MTCNN(image_size=160, select_largest=False, margin=20, min_face_size=10,
                      post_process=True, thresholds=[0.8, 0.9, 0.9])

        face_in_image = mtcnn(image)
        if face_in_image is None: # if no face was detected by MTCNN
            return None

        return face_in_image

    def __repr__(self):
        return self.__class__.__name__ + '()'
