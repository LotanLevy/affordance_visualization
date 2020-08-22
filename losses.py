
import numpy as np
import tensorflow as tf

class FeaturesLoss:
    def __init__(self, templates_images, model):
        self.templates_features = self.build_templates(templates_images, model)

    def build_templates(self, templates_images, model):
        templates = []
        for i in range(templates_images.shape[0]):
            image = np.expand_dims(templates_images[i], axis=0)
            templates.append(
                np.squeeze(model(image, training=False), axis=0))
        return np.array(templates)

    def __call__(self, labels, preds):
        preds_num = preds.shape[0]
        losses = np.zeros(preds_num)
        for i in range(preds_num):
            distances = []
            for t in range(self.templates_features.shape[0]):
                distances.append(np.sqrt(float(np.dot(preds[i] - self.templates_features[t],
                                                      preds[i] - self.templates_features[t]))))  # Eucleaden distance
            losses[i] = min(distances)
        return losses
