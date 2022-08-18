import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(i, y):
    plt.figure(1)
#     plt.clf() 此时不能调用此函数，不然之前的点将被清空。
    plt.subplot(111)
    plt.plot(i, y, '.')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
x = np.linspace(-10,10,500)
y = []
for i in range(len(x)):
    y = i
    y1 = np.cos(i/(3*3.14))
    y2 = np.sin(i/(3*3.14))
#     y.append(np.array([y1,y2])) #保存历史数据
    plot_durations(i, y)
