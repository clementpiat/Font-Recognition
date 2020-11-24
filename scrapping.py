from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import json
from time import time
import os

with open('config.json') as json_file:
    config = json.load(json_file)

driver = webdriver.Chrome()

fonts = ["Roboto"]
css_selector = '.variant__font-render'

for font in fonts:
    folder_path = f"{config['data_path']}/{font}"

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    driver.get(f'https://fonts.google.com/specimen/{font}')

    try:
        element_present = EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
        WebDriverWait(driver, config["timeout"]).until(element_present)
    except TimeoutException:
        print("Timed out waiting for page to load")
    finally:
        print("Page loaded")
        element = driver.find_elements_by_css_selector(css_selector)[0]
        element.screenshot(f"{folder_path}/{time()}.png")

driver.close()