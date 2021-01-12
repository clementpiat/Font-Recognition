from PIL import Image  
import random as rd
import numpy as np

class ImageTransformerRSR():
    def __init__(self, width=512, height=64, random_state=0, translation_ratio_x=0.7, translation_ratio_y=0.2):
        """
        square_dim: dimension of the final square image (number of pixels)
        """
        self.width = width
        self.height = height
        self.random_state = random_state
        rd.seed(random_state)
        self.translation_ratio_x = translation_ratio_x
        self.translation_ratio_y = translation_ratio_y

    def set_random_transformation_parameters(self):
        """
        angle: angle of rotation of the sentence (degrees)
        size: proportion of the text in the final image (float between 0.5 and 1)
        proportion_w, proportion_h: location of the text in the final image (couple of floats between 0 and 1)
        """
        self.angle = rd.randint(-4,4)
        self.resize_factor = rd.uniform(0.6,1.2)
        self.translation_x = (rd.random()*2 -1)*self.width*self.translation_ratio_x
        self.translation_y = (rd.random()*2 -1)*self.height*self.translation_ratio_y

    def __call__(self, img):
        """
        Being given a PIL image returns a rotated, shifted and resized image
        """
        self.set_random_transformation_parameters()
        # Rotate
        img = img.rotate(self.angle, expand=True, fillcolor='black')

        # Resize
        img_w, img_h = img.size
        new_shape = int(img_w*self.resize_factor), int(img_h*self.resize_factor)
        img = img.resize(new_shape)

        # Place it somewhere in a bigger white image
        background = Image.new('L', (self.width, self.height) , 0)
        offset = (int(self.translation_x), int(self.translation_y))
        background.paste(img, offset)
        return background