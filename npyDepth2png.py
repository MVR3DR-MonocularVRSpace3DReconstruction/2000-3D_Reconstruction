import os
from pathlib import Path
from tqdm import tqdm
import glob
import numpy as np
from PIL import Image

dep = Image.open("data/redwood-livingroom/depth/00000.png")
print(np.array(dep))
print(dep.getbands())
input()

data_dir = Path("data/classroom/")
os.system("rm -rf {}".format(str(data_dir/"depth_out")))
os.system("mkdir {}".format(str(data_dir/"depth_out")))

depth_files = sorted(glob.glob(str(data_dir/"depth_raw/*.npy")))
print(len(depth_files))
# groups = np.load(depth_files[0])
for idx in tqdm(range(len(depth_files))):
    img = np.load(depth_files[idx])
    png = np.uint8(img * 255)
    png = Image.fromarray(png,'L')
    
    png.save(str(data_dir/"depth_out/{:0>5}.png".format(idx)),format="png",quality = 100)
