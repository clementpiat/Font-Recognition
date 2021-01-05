import os

from torch.utils.data import Dataset

from .image_transformer import ImageTransformer
from .utils import get_n_pairs


class FontDataset(Dataset):
    def __init__(self, dataset_name, siamese=False, k_siamese=2, random_state=0, width=512, height=64, translation_ratio=0.05):
        super(FontDataset, self).__init__()

        self.dataset_name = dataset_name
        fonts = os.listdir(os.path.join("data", dataset_name))
        self.font_to_label = { font: i for i,font in enumerate(fonts) }
        self.filenames = []
        self.fonts = []
        for font in fonts:
            filenames = os.listdir(os.path.join("data", dataset_name, font))
            self.filenames += filenames
            self.fonts += [font] * len(filenames)

        self.it =  ImageTransformer(random_state=random_state, width=width, height=height, translation_ratio=translation_ratio)
        
        self.siamese = siamese
        self.k_siamese = k_siamese
        if self.siamese:
            self.indexes = get_n_pairs(k_siamese*len(self.filenames), self.fonts)
        

    def __getitem__(self, index):
        if self.siamese:
            index1, index2, label = self.indexes[index]
            img1 = self.get_img_label(index1)
            img2 = self.get_img_label(index2)
            return (*img1, *img2, label)
        else:
            return self.get_img_label(index)

    def __len__(self):
        if self.siamese:
            return self.k_siamese*len(self.filenames)
        else:
            return len(self.filenames)

    def get_img_label(self, index):
        filename = os.path.join("data", self.dataset_name, self.fonts[index], self.filenames[index])
        img = self.it.sentence_to_image(filename)

        return (img, self.font_to_label[self.fonts[index]])
