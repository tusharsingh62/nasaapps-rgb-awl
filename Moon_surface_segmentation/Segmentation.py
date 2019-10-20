from keras.models import Model,load_model
import cv2 as cv
import os
import matplotlib.pyplot as plt


TransferLearningModel = load_model('/Users/tushar/Downloads/model_TL_UNET.h5') # Load trained segmentation model


#InputPath = "/Users/tushar/Downloads/real_moon_images/"
img_x = cv.imread("PCAM5.png")                           # Path of image
img_x = cv.cvtColor(img_x, cv.COLOR_BGR2RGB)
img_x = cv.resize(img_x,(500,500))
img_x = img_x.reshape(1,500,500,3)

prediction = TransferLearningModel.predict(img_x)

pred = prediction.reshape(500,500,3)
pred_ = cv.resize(pred,(700,450))
plt.imshow(pred_)

