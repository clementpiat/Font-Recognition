from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from essential_generators import DocumentGenerator

import json
from time import time
import os
from tqdm import tqdm

with open('config.json') as json_file:
    config = json.load(json_file)

with open(config["fonts_file"]) as json_file:
    fonts = json.load(json_file)

gen = DocumentGenerator()
# driver = webdriver.Chrome()
driver = webdriver.Chrome("/home/rockl33/Code/chromedriver")

css_selector = '.variant__font-render'

for font in tqdm(fonts):
    folder_path = f"{config['data_path']}/{font}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    n_files = len(os.listdir(folder_path))

    # At the end we want exactly <number_of_sentences_per_font> images per font
    for _ in range(config["number_of_sentences_per_font"] - n_files):
        text = gen.sentence()
        driver.get(f'https://fonts.google.com/specimen/{font}?preview.text={text}&preview.text_type=custom')

        try:
            element_present = EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
            WebDriverWait(driver, config["timeout"]).until(element_present)
        except TimeoutException:
            print("Timed out waiting for page to load")
        finally:
            # print("Page loaded")
            # Additional safety
            try:
                element = driver.find_elements_by_css_selector(css_selector)[0]
                element.screenshot(f"{folder_path}/{time()}.png")
            except:
                print("Error while trying to screenshot...")

driver.close()