import os
import random

from torch.utils.data import Dataset

from .image_transformer import ImageTransformer


class FontDataset(Dataset):
    def __init__(self, dataset_name, font_to_filenames, siamese=False, n_transformations=2, random_state=0, width=512, height=64, translation_ratio=0.05):
        super(FontDataset, self).__init__()
       
        self.dataset_name = dataset_name
        self.font_to_label = { font: i for i,font in enumerate(font_to_filenames) }
        self.filenames, self.fonts, self.font_to_indexes = [], [], {}
        i = 0
        for font,filenames in font_to_filenames.items():
            n = len(filenames)

            self.filenames += filenames
            self.fonts += [font] * n
            self.font_to_indexes[font] = [j for j in range(i,i+n)]
            i += n

        self.n_filenames = i
        self.it =  ImageTransformer(random_state=random_state, width=width, height=height, translation_ratio=translation_ratio)
        self.siamese = siamese
        self.n_transformations = n_transformations

    
    def get_random_pair(self):
        is_positive_pair = random.random() > 0.5
        if is_positive_pair:
            font = random.choice(list(self.font_to_indexes.keys()))
            index1, index2 = random.choices(self.font_to_indexes[font], k=2)
        else:
            font1, font2 = random.choices(list(self.font_to_indexes.keys()), k=2)
            index1, index2 = random.choice(self.font_to_indexes[font1]), random.choice(self.font_to_indexes[font2])
        return index1,index2,is_positive_pair
        

    def __getitem__(self, index):
        if self.siamese:
            index1, index2, label = self.get_random_pair()
            img1 = self.get_img_label(index1)
            img2 = self.get_img_label(index2)
            return (*img1, *img2, label)
        else:
            return self.get_img_label(index%self.n_filenames)

    def __len__(self):
        return self.n_transformations * self.n_filenames

    def get_img_label(self, index):
        filename = os.path.join("data", self.dataset_name, self.fonts[index], self.filenames[index])
        img = self.it.sentence_to_image(filename)

        return (img, self.font_to_label[self.fonts[index]])

def get_train_test_dataset(dataset_name, train_size, siamese=False, n_transformations=2, random_state=0, width=512, height=64, translation_ratio=0.05):
    fonts = os.listdir(os.path.join("data", dataset_name))
    font_to_filenames_train, font_to_filenames_test = {}, {}
    for font in fonts:
        filenames = os.listdir(os.path.join("data", dataset_name, font))
        n = int(len(filenames)*train_size)

        random.shuffle(filenames) # Not necessary
        font_to_filenames_train[font] = filenames[:n]
        font_to_filenames_test[font] = filenames[n:]
    
    train_dataset = FontDataset(dataset_name, font_to_filenames_train, siamese=siamese, n_transformations=n_transformations, 
        random_state=random_state, width=width, height=height, translation_ratio=translation_ratio)
    test_dataset = FontDataset(dataset_name, font_to_filenames_test, siamese=siamese, n_transformations=n_transformations, 
        random_state=random_state, width=width, height=height, translation_ratio=translation_ratio)
    return train_dataset, test_dataset