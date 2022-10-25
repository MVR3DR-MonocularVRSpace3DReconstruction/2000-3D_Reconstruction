import cv2
import numpy as np
import glob
from tqdm import tqdm
img_array = []
img_dir = sorted(glob.glob("data/redwood-boardroom/image/*.jpg"))
for filename in img_dir: # [:len(img_dir)//8]
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


out = cv2.VideoWriter('boardroom.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in tqdm(range(len(img_array))):
    out.write(img_array[i])
out.release()