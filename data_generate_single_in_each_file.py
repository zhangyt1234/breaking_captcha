import string
from captcha.image import ImageCaptcha
import pandas as pd
import numpy as np

from PIL import Image

### Image Setup
image = ImageCaptcha(fonts = [r".\fonts\lato\Lato-Regular.ttf",
                              r".\fonts\lato\Lato-Bold.ttf",
                              r".\fonts\lato\Lato-Light.ttf",
                              r".\fonts\quicksand\Quicksand-Regular.otf",
                              r".\fonts\quicksand\Quicksand-Bold.otf",
                              r".\fonts\quicksand\Quicksand-Light.otf",
                              r".\fonts\open-sans\OpenSans-Regular.ttf",
                              r".\fonts\open-sans\OpenSans-Bold.ttf",
                              r".\fonts\open-sans\OpenSans-Light.ttf",
                              r".\fonts\raleway\Raleway-Bold.ttf",
                              r".\fonts\raleway\Raleway-Light.ttf",
                              r".\fonts\raleway\Raleway-regular.ttf",
                              r".\fonts\vera\Vera.ttf",
                              r".\fonts\vera\VeraBd.ttf",
                              ], width = 64, height = 64)


# Create Folders
import os
import time

start_time = time.time()
print ("Code start ...")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# Generate Data

times = 1                                   # number of pictures we want to generate for a single character

letters_total = string.ascii_letters[:]

letter_lower_capital = string.ascii_lowercase[:]
letter_upper_capital = string.ascii_uppercase[:]

count = 1
save_count = []
save_char = []


for number in range(0, 10):
    createFolder('./images/number/' + str(number))
    folder_link = './images/number/' + str(number) + '/'
    for i in range (1, times+1):
        image.write(str(number),  folder_link + str(count) + '.png')
        count = count + 1




for letter in letter_lower_capital:

    createFolder('./images/letter_lower/' + str(letter))
    folder_link = './images/letter_lower/' + str(letter) + '/'

    print(folder_link)
    for i in range (1, times+1):
        image.write(str(letter), folder_link + str(count) + '.png')

        count = count + 1


for letter in letter_upper_capital:

    createFolder('./images/letter_upper/' + str(letter).upper())
    folder_link = './images/letter_upper/' + str(letter).upper() + '/'

    print (folder_link)
    for i in range (1, times+1):
        image.write(str(letter), folder_link + str(count) + '.png')

        count = count + 1