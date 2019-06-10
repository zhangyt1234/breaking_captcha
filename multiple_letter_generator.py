import string
from captcha.image import ImageCaptcha
import random
import pandas as pd
import numpy as np

from PIL import Image

### Image Setup
image = ImageCaptcha(fonts = [r'/Users/yutong/Documents/Stanford/CS230/open-sans/OpenSans-Bold.ttf'], width = 64*2, height = 64)


# Create Folders
import os
import time

start_time = time.time()
print ("Code start ...")

num_images = 10000

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

createFolder('./111images_4_digits/')


# Generate Data

# character
letters_total = string.ascii_letters[:]

num_list = []

for x in range(0,10):
    num_list.append(str(x))

letter_list = []

for letter in letters_total:
    letter_list.append(letter)

all_char_list = num_list + letter_list

length_all_char_list = len(all_char_list)


count = 1
save_count = []
save_char = []

for i in range(0, num_images):
    digit_1 = random.randrange(0, length_all_char_list, 1)
    digit_2 = random.randrange(0, length_all_char_list, 1)
    digit_3 = random.randrange(0, length_all_char_list, 1)
    digit_4 = random.randrange(0, length_all_char_list, 1)
    text_char = all_char_list[digit_1] + all_char_list[digit_2] + all_char_list[digit_3] + all_char_list[digit_4]
    text_char_str = str(text_char)
    # image.write(text_char_str, './images_4_digits/' + str(count) + '.png')
    image.write(text_char_str, './111images_4_digits/' + text_char_str + '.png')
    save_count.append(count)
    save_char.append(text_char_str)
    count = count + 1

data = pd.DataFrame({"count" : save_count, "character" : save_char})

header = ["count", 'character']

data.to_csv('data_4_digits.csv', encoding = 'utf-8', columns = header)

end_time = time.time()





