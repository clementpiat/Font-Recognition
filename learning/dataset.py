import os

from torch.utils.data import Dataset

from .image_transformer import ImageTransformer


class FontDataset(Dataset):
    def __init__(self, dataset_name, random_state=0, width=512, height=64, translation_ratio=0.05):
        super(FontDataset, self).__init__()

        self.dataset_name = dataset_name
        fonts = os.listdir(os.path.join("data", dataset_name))
        self.font_to_label = dict([[font,i] for i,font in enumerate(fonts)])
        self.filenames = []
        self.fonts = []
        for font in fonts:
            filenames = os.listdir(os.path.join("data", dataset_name, font))
            self.filenames += filenames
            self.fonts += [font] * len(filenames)

        self.it =  ImageTransformer(random_state=random_state, width=width, height=height, translation_ratio=translation_ratio)
        

    def __getitem__(self, index):
        filename = os.path.join("data", self.dataset_name, self.fonts[index], self.filenames[index])
        img = self.it.sentence_to_image(filename)

        return (img, self.font_to_label[self.fonts[index]])

    def __len__(self):
        return len(self.filenames)