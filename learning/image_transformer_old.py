from PIL import Image  
import random as rd
import numpy as np

class ImageTransformer():
    def __init__(self, width=512, height=64, random_state=0, translation_ratio=0.05):
        """
        square_dim: dimension of the final square image (number of pixels)
        """
        self.width = width
        self.height = height
        self.random_state = random_state
        rd.seed(random_state)
        self.translation_ratio = translation_ratio

    def set_random_transformation_parameters(self):
        """
        angle: angle of rotation of the sentence (degrees)
        size: proportion of the text in the final image (float between 0.5 and 1)
        proportion_w, proportion_h: location of the text in the final image (couple of floats between 0 and 1)
        """
        self.angle = rd.randint(-2,2)
        self.resize_factor = rd.uniform(0.8,1.2)
        self.translation_x = rd.random()*self.width*self.translation_ratio
        self.translation_y = rd.random()*self.height*self.translation_ratio

    def sentence_to_image(self, path_to_sentence):
        """
        Being given the path to a font image returns a rotated, shifted and resized image
        """
        self.set_random_transformation_parameters()

        img = Image.open(path_to_sentence, 'r')

        # Rotate
        img = img.rotate(self.angle, expand=True, fillcolor='white')

        # Resize
        img_w, img_h = img.size
        new_shape = int(img_w*self.resize_factor), int(img_h*self.resize_factor)
        img = img.resize(new_shape)

        # Place it somewhere in a bigger white image
        background = Image.new('L', (self.width, self.height) , 255)
        offset = (int(self.translation_x), int(self.translation_y))
        background.paste(img, offset)
        return np.array(background)[np.newaxis,:,:]