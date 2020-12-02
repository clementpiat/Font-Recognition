import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

from learning.dataset import FontDataset

dataset = FontDataset("10_fonts_100_sentences")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for data in dataloader:
    img, label = data
    break

print(label)
plt.imshow(np.array(img[0]))
plt.show()