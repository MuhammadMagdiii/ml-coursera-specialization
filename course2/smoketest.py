import graphlab as gl
import numpy as np
import os

# set working directory
os.chdir('/Users/WKT/Projects/ML-coursera')

# numpy example
a = np.arange(15).reshape(3, 5)

# graphlab example
products = gl.SFrame('amazon_baby.gl/')


def house(sf):
    return (280.76 * sf - 44850)

def house2(sm):
    return (3022.076789769975 * sm - 44850)

