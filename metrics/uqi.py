from sewar.full_ref import uqi, mse, psnr, ssim
import sys
import cv2

# 3. Load the two input images
imageA = cv2.imread(sys.argv[1])
imageB = cv2.imread(sys.argv[2])

# # 4. Convert the images to grayscale
# grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

print('uqi', uqi(imageA,imageB))
