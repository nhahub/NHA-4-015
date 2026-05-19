from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

class EnsembleModel:
    def __init__(self, model_paths):
        self.models = [load_model(path) for path in model_paths]

    def _augment_image(self, img):
        return [
            img,
            img.transpose(Image.FLIP_LEFT_RIGHT),
            img.transpose(Image.FLIP_TOP_BOTTOM),
            img.rotate(15),
            img.rotate(-15)
        ]

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        all_preds = []

        for model in self.models:
            input_shape = model.input_shape[1:3]
            resized = img.resize(input_shape)
            aug_imgs = self._augment_image(resized)
            preds = []

            for aug in aug_imgs:
                arr = image.img_to_array(aug) / 255.0
                arr = np.expand_dims(arr, axis=0)
                preds.append(model.predict(arr)[0])

            model_avg = np.mean(preds, axis=0)
            all_preds.append(model_avg)

        return np.mean(all_preds, axis=0)