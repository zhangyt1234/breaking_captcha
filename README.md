### Before you get started

source code based on https://github.com/rom1504/tensorflow_captcha_solver

### New work compared to source code

Use new segmentation algorithm which combine the pixel density and even-distance segmentation method.
Train with LeNet-5 model.

### Installation
To run these scripts, you need the following installed:

1. Python 3
2. The python libraries listed in requirements.txt
 - virtualenv --python=/usr/bin/python3 venv
 - source venv/bin/activate
 - pip3 install -r requirements.txt

installing `msttcorefonts` provide more fonts hence better results
(that was useful in my case to distinguish O and 0)

Run pipeline.sh or :
 
### Step 0: Generate images

python3 multiple_letter_generator.py

### Step 1: Extract single letters from CAPTCHA images

Run:

python3 own_segmentation.py

The results will be stored in the "extracted_letter_images" folder.


### Step 2: Train the neural network to recognize single letters

Run:

python3 LeNet-5_train_model.py

This will write out "captcha_model.hdf5" and "model_labels.dat"


### Step 3: Use the model to solve CAPTCHAs!

Run: 

python3 solve_captchas_with_model.py


