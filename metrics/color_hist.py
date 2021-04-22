"""
 * Python program to create a color histogram.
 *
 * Usage: python ColorHistogram.py <filename>
"""
import sys
import skimage.io
import numpy as np
import skimage.viewer
from matplotlib import pyplot as plt

# read original image, in full color, based on command
# line argument
image = skimage.io.imread(fname=sys.argv[1])

# # display the image
# viewer = skimage.viewer.Viewer(image)
# viewer.show()


# tuple to select colors of each channel line
colors = ("r", "g", "b")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.xlabel("Color value")
plt.ylabel("Pixels")

plt.show()


# % OUTPUT:
# %        results.npcr_score: quantitative NPCR score (larger is better)
# %        results.npcr_pVal : qualitative NPCR score  (larger is better)
# %        results.npcr_dist : theoretical NPCR normal dist. (mean +\- var)
# %        results.uaci_score: quantitative UACI score (larger is NOT better)
# %        results.uaci_pVal : qualitative UACI score  (larger is better)
# %        results.uaci_dist : theoretical UACI normal dist. (mean +\- var)
# % =========================================================================