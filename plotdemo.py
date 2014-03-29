#!/bin/env python
#-*- encoding:utf-8 -*-

#http://matplotlib.org/api/pyplot_api.html
#没有解决多个图像间的间距，其他都ok

# import sys
# import matplotlib.pyplot as plt
# import pylab

# plt.subplots(4,1)
# plt.figure(1)
# plt.tight_layout(pad=2)
# plt.subplot(411)
# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# # #plt.show()
# plt.subplot(412)
# #plt.plot([1,2,3,4], [1,4,9,16],figure = fig)
# plt.plot([1,2,3,4], [1,4,9,16])
# plt.subplot(413)
# import numpy as np
# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)

# # the histogram of the data
# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)

# plt.subplot(414)
# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)

# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

# #X = 10*np.random.rand(5,3)
# #plt.imshow(X,aspect="auto")
# #plt.show()

# # 用dpi来控制图像大小和质量
# pylab.savefig('foo.png',bbox_inches="tight",dpi=600)
# #pylab.savefig('foo.png')

# from __future__ import print_function
# """
# A very simple 'animation' of a 3D plot
# """
# from mpl_toolkits.mplot3d import axes3d
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# def generate(X, Y, phi):
#     R = 1 - np.sqrt(X**2 + Y**2)
#     return np.cos(2 * np.pi * X + phi) * R

# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# xs = np.linspace(-1, 1, 50)
# ys = np.linspace(-1, 1, 50)
# X, Y = np.meshgrid(xs, ys)
# Z = generate(X, Y, 0.0)

# wframe = None
# tstart = time.time()
# for phi in np.linspace(0, 360 / 2 / np.pi, 100):

#     oldcol = wframe

#     Z = generate(X, Y, phi)
#     wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)

#     # Remove old line collection before drawing
#     if oldcol is not None:
#         ax.collections.remove(oldcol)

#     plt.draw()

# print ('FPS: %f' % (100 / (time.time() - tstart)))

#########################################

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np

# n_angles = 36
# n_radii = 8

# # An array of radii
# # Does not include radius r=0, this is to eliminate duplicate points
# radii = np.linspace(0.125, 1.0, n_radii)

# # An array of angles
# angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

# # Repeat all angles for each radius
# angles = np.repeat(angles[...,np.newaxis], n_radii, axis=1)

# # Convert polar (radii, angles) coords to cartesian (x, y) coords
# # (0, 0) is added here. There are no duplicate points in the (x, y) plane
# x = np.append(0, (radii*np.cos(angles)).flatten())
# y = np.append(0, (radii*np.sin(angles)).flatten())

# # Pringle surface
# z = np.sin(-x*y)

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)

# plt.show()

###############################################
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

#bx = fig.add_subplot(212, projection="3d")
X, Y, Z = axes3d.get_test_data(1)
print '========================'
#print X.dumps()
print X.ndim
print X.size
print X.flat
print X.shape
print X.dtype
print type(X)
print type(X[0])
print X
newX = X.transpose()
print X.transpose()
print newX == Y
print "======================="
print Y.shape
print Y
print "======================="
print Z
print type(Z)
X2,Y2,Z2 = axes3d.get_test_data(0.2)
import matplotlib.colors as colors
converter = colors.ColorConverter()
red = converter.to_rgb("r")
colorvar = converter.to_rgb('0.8')
print "red ",red

from pylab import *
NUM_COLORS = 22
cm = get_cmap('gist_rainbow')
colors = []
for i in range(NUM_COLORS):
    color0 = cm(1.*i/NUM_COLORS)  # color will now be an RGBA tuple
    colors.append(color0)
    print color0
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1,color=colors[14])
#ax.plot_wireframe(X2, Y2, Z2, rstride=10, cstride=10,color=colors[1])

#bx.plot_wireframe(X2,Y2,Z2,rstride=10,cstride=10)

plt.show()