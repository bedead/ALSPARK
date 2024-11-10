from PIL import Image
import cv2
import numpy as np


def is_high_resolution(
    image: Image.Image, min_resolution=(480, 320), sharpness_threshold=60.0
) -> int:
    """
    Determines if an image has "good" or "low" resolution based on pixel density and sharpness.

    Parameters:
    - image (PIL.Image): PIL Image object.
    - min_resolution (tuple): Minimum (width, height) resolution for an image to be considered "good".
    - sharpness_threshold (float): Minimum sharpness score (variance of Laplacian) for an image to be considered "sharp".

    Returns:
    - int: "Good resolution" = 1 or "Low resolution" = 0 based on the checks.
    """
    width, height = image.size
    if width < min_resolution[0] or height < min_resolution[1]:
        return 0

    image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    if laplacian_var < sharpness_threshold:
        return 0

    return 1


# Example usage
# image = Image.open("path/to/your/image.jpg")
# result = is_high_resolution(image)
# print(result)
