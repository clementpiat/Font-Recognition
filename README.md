# Font Identification

> CentraleSupélec Deep Learning Project

Font identification using a CNNs siamese network.

### Features

- Creation of a font dataset by scraping Google Fonts and Dafont.
- Training of a siamese model that returns a similarity out of two input images. The similarity is high when the same font is used.

## Environment

Use `conda` and the `environment.yml` file at the root of the project to install the dependencies.

## Creating a dataset

You can create your own `scraping/config.json` file or keep the default one.

Once you have chosen the config, please run `cd scraping` and `python scraping_dafont.py` (Dafont) or `python scraping.py` (Google Fonts).

For this latter, a ChromeDriver is required.

## Training a model

Choose your arguments and run `python training.py [args]` or use the default one `python training.py`.

You can have information about the parameters by running `python training.py -h`

## Fine tuning a model

Choose your arguments and run `python hyperoptimization.py [args]` or use the default one `python hyperoptimization.py`.

You can have information about the parameters by running `python hyperoptimization.py -h`

## Saving the model and its results

The model and its results are automatically saved on a dedicated folder in the `result` folder.

Some Notebooks are available at the root of the project and in `test_scenario` to plot results and see the model behaviour.
