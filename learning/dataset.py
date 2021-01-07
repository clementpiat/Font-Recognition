import os
import random

from torch.utils.data import Dataset

from .image_transformer import ImageTransformer


class FontDataset(Dataset):
    def __init__(self, dataset_name, font_to_label, font_to_filenames, it, n_transformations, siamese=False):
        super(FontDataset, self).__init__()
       
        self.dataset_name = dataset_name
        self.font_to_label = font_to_label
        self.filenames, self.fonts, self.font_to_indexes = [], [], {}
        i = 0
        for font,filenames in font_to_filenames.items():
            n = len(filenames)

            self.filenames += filenames
            self.fonts += [font] * n
            self.font_to_indexes[font] = [j for j in range(i,i+n)]
            i += n

        self.n_filenames = i
        self.siamese = siamese
        self.it =  it
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
        return (self.it(self.filenames[index]), self.font_to_label[self.fonts[index]])

def get_train_test_dataset(dataset_name, train_size, n_transformations, siamese=False, random_state=0, width=512, max_height_ratio=0.4, max_width_ratio=0.15):
    fonts = os.listdir(os.path.join("data", dataset_name))
    font_to_filenames_train, font_to_filenames_test, paths_filenames = {}, {}, []
    for font in fonts:
        font_path = os.path.join("data", dataset_name, font)
        filenames = os.listdir(font_path)
        paths_filenames += [(os.path.join(font_path, filename), filename) for filename in filenames]
        n = int(len(filenames)*train_size)

        font_to_filenames_train[font] = filenames[:n]
        font_to_filenames_test[font] = filenames[n:]

    font_to_label = { font: i for i,font in enumerate(font_to_filenames_train) }

    it = ImageTransformer(random_state=random_state, width=width, max_width_ratio=max_width_ratio, max_height_ratio=max_height_ratio)
    it.load_and_crop(paths_filenames)
    
    train_dataset = FontDataset(dataset_name, font_to_label, font_to_filenames_train, it, n_transformations, siamese=siamese)
    test_dataset = FontDataset(dataset_name, font_to_label, font_to_filenames_test, it, n_transformations, siamese=siamese)
    return train_dataset, test_dataset