import os
import cv2

input_dir = "C:/Users/91879/Desktop/Project/ImageProject/celeba" 
output_dir = "C:/Users/91879/Desktop/Project/ImageProject/low"

low_res_size = (22, 22)
b=os.listdir(input_dir)
for filename in b:
    img = cv2.imread(os.path.join(input_dir, filename))
    img = cv2.resize(img, low_res_size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(output_dir, filename), img)