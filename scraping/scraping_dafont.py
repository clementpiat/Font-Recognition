import requests
import shutil
from essential_generators import DocumentGenerator
import json
import os
from time import time
from tqdm import tqdm

try:
    with open('config.json') as json_file:
        config = json.load(json_file)
except:
    with open('default_config.json') as json_file:
        config = json.load(json_file)

with open(config["dafont_file"]) as json_file:
    fonts = json.load(json_file)

gen = DocumentGenerator()

if not os.path.exists(config['data_path']):
    os.mkdir(config['data_path'])

for font, ext in tqdm(fonts):
    folder_path = f"{config['data_path']}/{font}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    n_files = len(os.listdir(folder_path))

    # At the end we want exactly <number_of_sentences_per_font> images per font
    for _ in range(config["number_of_sentences_per_font"] - n_files):
        text = gen.sentence()
        url = f'https://img.dafont.com/preview.php?text={text}&ttf={font}&ext={ext}&size=50&psize=m&y=60)'
        response = requests.get(url, stream=True)
        with open(f"{folder_path}/{time()}.png", 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response