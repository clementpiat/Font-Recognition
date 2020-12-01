from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from essential_generators import DocumentGenerator

import json
from time import time, sleep
import os
from tqdm import tqdm

with open('scraping_config.json') as json_file:
    config = json.load(json_file)

with open(config["fonts_file"]) as json_file:
    fonts = json.load(json_file)

gen = DocumentGenerator()
if not config["chromedriver_path"]:
    driver = webdriver.Chrome()
else:
    driver = webdriver.Chrome(config["chromedriver_path"])

font_selector = '.variant__font-render'
style_selector = '.variant__style'
style = 'Regular 400'
sleep_time = 0.5 # Time of sleeping between scrolling and screenshot (in seconds).

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
            # Wait for the first font style to be loaded, we assume that every style load almost at the same time.
            element_present = EC.presence_of_element_located((By.CSS_SELECTOR, font_selector))
            WebDriverWait(driver, config["timeout"]).until(element_present)
        except TimeoutException:
            print("Timed out waiting for page to load")
        finally:
            # print("Page loaded")
            # Additional safety
            try:
                elements = driver.find_elements_by_css_selector(".variant")
                styles = [el.find_element_by_css_selector(style_selector).text for el in elements]
                if style in styles:
                    style_index = styles.index(style)
                else:
                    break

                element = driver.find_elements_by_css_selector(font_selector)[style_index]
                # Take the screenshot
                height = driver.get_window_size()["height"]
                driver.execute_script(f"window.scrollTo(0, {element.location['y'] - height//2});")
                sleep(sleep_time)
                element.screenshot(f"{folder_path}/{time()}.png")  
            except:
                print("Error while trying to screenshot...")

driver.close()