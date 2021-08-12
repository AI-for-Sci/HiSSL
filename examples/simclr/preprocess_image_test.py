import numpy as np
from hissl.utils.data_util import preprocess_image

if __name__ == '__main__':
    image = np.random.random((28, 28, 3))
    height = 28
    width = 28
    image = preprocess_image(image, height, width, is_training=False,
                             color_distort=True, test_crop=True)
    print("image: ", image)
