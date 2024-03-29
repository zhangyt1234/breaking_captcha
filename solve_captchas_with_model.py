from imutils import paths
import numpy as np

from captcha_solver import solve_captcha, load_captcha_model


# MODEL_FILENAME = "captcha_model.hdf5"
# MODEL_LABELS_FILENAME = "model_labels.dat"
# CAPTCHA_IMAGE_FOLDER = "annotated"

MODEL_FILENAME = "captcha_model3.hdf5"
MODEL_LABELS_FILENAME = "model_labels3py.dat"
CAPTCHA_IMAGE_FOLDER = "tested"

model = load_captcha_model(MODEL_FILENAME, MODEL_LABELS_FILENAME)


# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)


# loop over the image paths
for image_file in captcha_image_files:

    captcha_text = solve_captcha(image_file, model)

    if captcha_text == "":
        continue

    print("CAPTCHA text is: {} for {}".format(captcha_text, image_file))
