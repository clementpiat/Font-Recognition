from PIL import Image  
import random as rd

class ImageTransformer():
    def __init__(self, square_dim=500, random_state=0):
        """
        square_dim: dimension of the final square image (number of pixels)
        """
        self.square_dim = square_dim
        self.random_state = random_state
        rd.seed(random_state)

    def set_random_transformation_parameters(self):
        """
        angle: angle of rotation of the sentence (degrees)
        size: proportion of the text in the final image (float between 0.5 and 1)
        proportion_w, proportion_h: location of the text in the final image (couple of floats between 0 and 1)
        """
        self.angle = rd.randint(0,359)
        self.size = rd.uniform(0.5,1)
        self.proportion_w = rd.random()
        self.proportion_h = rd.random() 

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
        coefficient = self.square_dim * self.size / max(img.size) 
        new_img_w, new_img_h = int(img_w*coefficient), int(img_h*coefficient)
        img = img.resize((new_img_w, new_img_h))
        # Place it somewhere in a bigger white image
        background = Image.new('RGBA', (self.square_dim, self.square_dim) , (255, 255, 255, 255))
        offset = (int((self.square_dim - new_img_w)*self.proportion_w), int((self.square_dim - new_img_h)*self.proportion_h))
        background.paste(img, offset)
        return background