import os
import cv2
import numpy as np


DATA_DIR = "./data/room/"
COLOR_LIST = sorted(os.listdir(DATA_DIR+'image/'))
DEPTH_LIST = sorted(os.listdir(DATA_DIR+'depth/'))
STEP = 1


# target = generate_point_cloud(
# 	DATA_DIR+'image/'+COLOR_LIST[366],
# 	DATA_DIR+'depth/'+DEPTH_LIST[366]
# 	)
for i in range(len(COLOR_LIST)):
    image = cv2.imread(DATA_DIR+'depth/'+DEPTH_LIST[i])
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    print(image)
    # print([[int(pixel/2) for pixel in line] for line in image])

    image = np.asarray([[int(pixel/2) for pixel in line] for line in image])
    print(image)
    cv2.imwrite(DATA_DIR+'gray/'+DEPTH_LIST[i],image)