# yoda_chimpanzee
This was a project I made while messing around with torchvision. While I was using the ResNet50_Weights.IMAGENET1K_V1 model, I gave it a picture of Yoda and the model called him a chimpanzee. So, I decided to make my own model for the sole purpose of being able to tell the difference between Yoda and a chimpanzee.

The images used for the project are on my google drive. The testing and training image folders should go in the root directory of the project.
https://drive.google.com/drive/folders/1DRn2DLay2gO79l0b_4L6lDGMkgcDHI2X?usp=sharing

training.py will create a new model from images of Yoda and chimpanzees in training_images, printing out the running loss as it does so. The two labels the model will recognize are "chimpanzee" and "yoda".

testing.py uses the previously created model to identify ten test images of Yoda and chimpanzees. The test images located in testing_images are specifically named from 0.jpg to 9.jpg.

In my testing, the model will tend to get 8 or 9 of the 10 test images correct. I'm not a data scientist, so I'm not sure how to improve this. This was as far as I got with knowing python and reading tutorials.
