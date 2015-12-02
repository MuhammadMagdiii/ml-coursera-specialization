import graphlab as gl
import numpy as np
import os

# set working directory
os.chdir('/Users/WKT/Projects/ML-coursera')

# numpy example
a = np.arange(15).reshape(3, 5)

# graphlab example
products = gl.SFrame('amazon_baby.gl/')
