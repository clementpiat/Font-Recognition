from PIL import Image, ImageOps
import random as rd
import numpy as np
from .image_transformer_old import ImageTransformerRSR

class ImageTransformer():
    def __init__(self, width, max_height_ratio=0.4, max_width_ratio=0.15, random_state=0, rsr=True):
        self.w = width
        self.max_height_ratio = max_height_ratio
        self.max_width_ratio = max_width_ratio
        rd.seed(random_state)
        self.rsr = rsr
        
    def find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    def load_and_crop(self, paths_filenames):
        self.images = {}
        for path, filename in paths_filenames:
            img = Image.open(path).convert('RGB')
            img = ImageOps.invert(img).convert('L')
            self.h = img.size[1]
            self.images[filename] = img.crop((0, 0, self.w, self.h))

        if self.rsr:
            self.image_transformer_rsr = ImageTransformerRSR(width=self.w, height=self.h)
        
    def transform_perspective(self, img):
        side = rd.randint(1,4)
        
        if side % 2 == 0:
            perspective_delta = self.h*self.max_height_ratio*rd.random()**2
            start_h = perspective_delta*rd.random()
            if side == 2:
                pa = [(0, 0), (0, self.h), (self.w, start_h), (self.w, start_h + self.h - perspective_delta)]
            else:
                pa = [(0, start_h), (0, start_h + self.h - perspective_delta), (self.w, 0), (self.w, self.h)]
        else:
            perspective_delta = self.w*self.max_width_ratio*rd.random()**2
            start_w = perspective_delta*rd.random()
            if side == 1:
                pa = [(start_w, 0), (0, self.h), (start_w + self.w - perspective_delta, 0), (self.w, self.h)]
            else:
                pa = [(0, 0), (start_w, self.h), (self.w, 0), (start_w + self.w - perspective_delta, self.h)]
        coeffs = self.find_coeffs(pa, [(0, 0), (0, self.h), (self.w, 0), (self.w, self.h)])
        return img.transform((self.w, self.h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    
    def transform(self,img):
        if self.rsr:
            return self.image_transformer_rsr(self.transform_perspective(img))
        else:
            return self.transform_perspective(img)

    def __call__(self, filename):
        return np.array(self.transform(self.images[filename]))[np.newaxis,:,:]