import os.path
import cv2
import glob
import numpy as np
#from matplotlib import pyplot as plt

CAPTCHA_IMAGE_FOLDER = "111images_4_digits"
OUTPUT_FOLDER = "111own_extracted_single_letter_images"


def segment_image(image_file):

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    x = image.shape[0]  # length in first dimension
    y = image.shape[1]  # length in second dimensiony(image)

    background_pixel = image[0][0]
    threshold = 450
    column_pixels = [] # pixels sum of each column, 128 columns
    min_column_pixels = 64
    min_column_index = -1
    trunks = [] # 8 trunks evenly
    trunks_min = []
    trunk_count = 0

    for j in range(0,y):
        count = 0
        for i in range(0,x):
            if image[i][j] != background_pixel:
                count = count + 1

        column_pixels.append(count)
        if count < min_column_pixels:
            min_column_pixels = count
            min_column_index = j

        trunk_count += count

        if (j + 1) % 16 == 0:
            trunks.append(trunk_count)
            trunks_min.append(min_column_index)
            trunk_count = 0
            min_column_pixels = 64


    left_border = 0
    right_border = 0
    letter_image_regions = []
    (x0, y0, w, h) = (0, 0, 0, 0)
    indexs = []
    for index in range(0,8):

        if trunks[index] <= threshold:
            right_border = trunks_min[index]
            if right_border <= 8:
                left_border =  right_border
                x0 = left_border
                continue
            if left_border == right_border:
                continue
            w = (int)(right_border - left_border)
            (x0, y0, w, h) = (x0, 0, w, 64)
            letter_image_regions.append((x0, y0, w, h))
            x0 = x0 + w
            left_border = right_border
            indexs.append(index)


        else:
            continue

    while (len(letter_image_regions) > 4):

        min_i = -1
        two_sum_min = 900
        for i in range(0, len(letter_image_regions) - 1):
            two_sum = trunks[indexs[i]] + trunks[indexs[i + 1]]
            if two_sum < two_sum_min:
                two_sum_min = two_sum
                min_i = i

        if min_i >= 0:
            new_x = letter_image_regions[min_i][0]
            new_w = letter_image_regions[min_i][2] + letter_image_regions[min_i + 1][2]

            letter_image_regions[min_i] = (new_x, 0, new_w, 64)
            # indexs[min_i] = indexs[min_i + 1]

            if min_i + 1 == len(letter_image_regions) - 1:
                del letter_image_regions[-1]
                continue
            for start in range(min_i + 1, len(letter_image_regions) - 1):
                if start + 1 >= len(letter_image_regions) - 1:
                    del letter_image_regions[-1]
                    break
                letter_image_regions[start] = letter_image_regions[start + 1]
                indexs[start] = indexs[start + 1]
                del letter_image_regions[-1]

        else:
            break


    if len(letter_image_regions) == 3:
        max_index = 0
        max_trunk = trunks[indexs[0]]
        for i in range(1, 3):
            if trunks[indexs[i]] > max_trunk:
                max_trunk = trunks[indexs[i]]
                max_index = i
        new_x1 = letter_image_regions[max_index][0]
        new_x2 = (int)(letter_image_regions[max_index][0] + letter_image_regions[max_index][2] / 2)
        new_width = (int) (letter_image_regions[max_index][2] / 2)

        new_letter_image_regions = []

        if max_index == 0:
            new_letter_image_regions.append((new_x1, 0, new_width, 64))
            new_letter_image_regions.append((new_x2, 0, new_width, 64))
            new_letter_image_regions.append(letter_image_regions[1])
            new_letter_image_regions.append(letter_image_regions[2])

        if max_index == 1:
            new_letter_image_regions.append(letter_image_regions[0])
            new_letter_image_regions.append((new_x1, 0, new_width, 64))
            new_letter_image_regions.append((new_x2, 0, new_width, 64))
            new_letter_image_regions.append(letter_image_regions[2])

        if max_index == 2:
            new_letter_image_regions.append(letter_image_regions[0])
            new_letter_image_regions.append(letter_image_regions[1])
            new_letter_image_regions.append((new_x1, 0, new_width, 64))
            new_letter_image_regions.append((new_x2, 0, new_width, 64))

        letter_image_regions = new_letter_image_regions

    if len(letter_image_regions) == 2:
        max_index = 0
        max_trunk = trunks[indexs[0]]
        if trunks[indexs[1]] > max_trunk:
            max_trunk = trunks[indexs[1]]
            max_index = 1
        new_x1 = letter_image_regions[max_index][0]
        new_x2 = (int) (letter_image_regions[max_index][0] + letter_image_regions[max_index][2] / 3)
        new_x3 = (int) (letter_image_regions[max_index][0] + letter_image_regions[max_index][2] * 2 / 3)
        new_width = (int) (letter_image_regions[max_index][2] / 3)

        new_letter_image_regions = []
        if max_index == 0:
            new_letter_image_regions.append((new_x1, 0, new_width, 64))
            new_letter_image_regions.append((new_x2, 0, new_width, 64))
            new_letter_image_regions.append((new_x3, 0, new_width, 64))
            new_letter_image_regions.append(letter_image_regions[1])

        if max_index == 1:
            new_letter_image_regions.append(letter_image_regions[0])
            new_letter_image_regions.append((new_x1, 0, new_width, 64))
            new_letter_image_regions.append((new_x2, 0, new_width, 64))
            new_letter_image_regions.append((new_x3, 0, new_width, 64))

        letter_image_regions = new_letter_image_regions

    if len(letter_image_regions) == 1:
        new_x1 = letter_image_regions[0][0]
        new_x2 = (int) (letter_image_regions[0][0] + letter_image_regions[0][2] / 4)
        new_x3 = (int) (letter_image_regions[0][0] + letter_image_regions[0][2] / 2)
        new_x4 = (int) (letter_image_regions[0][0] + letter_image_regions[0][2] * 3 / 4)
        new_width = (int) (letter_image_regions[0][2] / 4)

        new_letter_image_regions = []
        new_letter_image_regions.append((new_x1, 0, new_width, 64))
        new_letter_image_regions.append((new_x2, 0, new_width, 64))
        new_letter_image_regions.append((new_x3, 0, new_width, 64))
        new_letter_image_regions.append((new_x4, 0, new_width, 64))

        letter_image_regions = new_letter_image_regions

    if len(letter_image_regions) == 0:
        letter_image_regions.append((0, 0, 32, 64))
        letter_image_regions.append((32, 0, 32, 64))
        letter_image_regions.append((64, 0, 32, 64))
        letter_image_regions.append((96, 0, 32, 64))
    #letter_image_regions = sorted(letter_image_regions, key=lambda y: y[0])
    '''letter_image_regions = []
    letter_image_regions.append((0, 0, 32, 64))
    letter_image_regions.append((32, 0, 32, 64))
    letter_image_regions.append((64, 0, 32, 64))
    letter_image_regions.append((96, 0, 32, 64))'''
    letter_images = []

    # loop over the letters
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        letter_image = image[y:y + h, x:x + w]

        letter_images.append(letter_image)

    return letter_images

captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    # our file name is not the label(correct_text) of the text
    captcha_correct_text = os.path.splitext(filename)[0]
    # captcha_correct_text = os.path.splitext(filename)[0]

    letter_images = segment_image(captcha_image_file)

    # Save out each letter as a single image
    for letter_image, letter_text in zip(letter_images, captcha_correct_text):
        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1

'''filename = os.path.basename(image_file)
# our file name is not the label(correct_text) of the text
captcha_correct_text = os.path.splitext(filename)[0]

letter_images = segment_image(image_file)

# Save out each letter as a single image
for letter_image, letter_text in zip(letter_images, captcha_correct_text):
            # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

            # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

            # write the letter image to a file
        # count = counts.get(letter_text, 1)
        # p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        p = os.path.join(save_path, "{}.png".format("0".zfill(6)))
        cv2.imwrite(p, letter_image)

            # increment the count for the current key
        # counts[letter_text] = count + 1
'''



