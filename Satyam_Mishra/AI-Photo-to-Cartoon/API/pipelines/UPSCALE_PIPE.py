import cv2
from PIL import Image
import numpy as np


def initialize_upscaler():
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("weights/FSRCNN_x2.pb")
    sr.setModel("fsrcnn", 2)

    return sr


def upscale(image, model):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = model.upsample(opencv_image)

    enhanced_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return enhanced_image
