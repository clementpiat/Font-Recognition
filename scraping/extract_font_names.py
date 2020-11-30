from selenium import webdriver
import json
import time
import argparse

def extract_font_names(n_fonts):
    driver = webdriver.Chrome()
    driver.get(f'https://fonts.google.com')

    all_fonts_selector = 'h1.mat-text--title'
    fonts = set()
    i = 0

    while len(fonts) < n_fonts:
        driver.execute_script(f"window.scrollTo(0, {i*200});")
        elements = driver.find_elements_by_css_selector(all_fonts_selector)
        for e in elements:
            try:
                fonts.add(e.get_attribute('textContent'))
            except:
                pass
        time.sleep(0.1)
        i += 1

    driver.close()

    with open(f'../font_names/fonts_{n_fonts}.json', 'w') as f:
        json.dump(list(fonts)[:n_fonts], f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_fonts", type=int, default=100)
    args = parser.parse_args()

    extract_font_names(args.n_fonts)

