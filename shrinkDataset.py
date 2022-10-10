import os
import glob
from tqdm import tqdm

shrink_ratio = 0.1
input_dir = "data/redwood-boardroom/"
output_dir = "data/shrink-redwood-boardroom/"

image_names = glob.glob(input_dir+"image/*.jpg")
depth_names = glob.glob(input_dir+"depth/*.png")

assert len(image_names) == len(depth_names)
n_images = len(image_names)
image_names = [image_names[idx] for idx in range(0,n_images,int(1/shrink_ratio))]
depth_names = [depth_names[idx] for idx in range(0,n_images,int(1/shrink_ratio))]

os.system("rm -rf {}".format(output_dir))
os.system("mkdir {}".format(output_dir))
os.system("mkdir {}depth/".format(output_dir))
os.system("mkdir {}image/".format(output_dir))

for idx in range(len(image_names)):
    # image_name = image_names[idx].split('/')[-1]
    # depth_name = depth_names[idx].split('/')[-1]
    os.system("cp {} {}".format(image_names[idx], output_dir+"image/"))
    os.system("cp {} {}".format(depth_names[idx], output_dir+"depth/"))