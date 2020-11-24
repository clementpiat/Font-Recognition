from selenium import webdriver
import json
from time import time
import os

with open('config.json') as json_file:
    config = json.load(json_file)

driver = webdriver.Chrome()

fonts = ["Roboto"]

for font in fonts:
    folder_path = f"{config['data_path']}/{font}"

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    driver.get(f'https://fonts.google.com/specimen/{font}')
    driver.save_screenshot(f"{folder_path}/{time()}.png")

driver.close()