import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
from math import sin, cos, pi
from mako.template import Template
from scipy import fftpack
import cv2
import sys
import tkinter as tk
from PIL import ImageDraw, Image, ImageTk
import sys

datal = cv2.imread(sys.argv[1] if len(sys.argv) >=2 else "DSC07693.png")/255
h, w = datal.shape[:2]

dh = h//16
dw = w//16

window = tk.Tk(className="bla")
original = Image.fromarray(np.round(datal[:,:,::-1]*255).astype(np.uint8))

canvas = tk.Canvas(window, width=original.size[0], height=original.size[1])
canvas.pack()
image_tk = ImageTk.PhotoImage(original)

canvas.create_image(original.size[0]//2, original.size[1]//2, image=image_tk)

db = datal[:,:,0]
dg = datal[:,:,1]
dr = datal[:,:,2]

dcb = fftpack.dct(fftpack.dct(db.T/w).T/h)
dcg = fftpack.dct(fftpack.dct(dg.T/w).T/h)
dcr = fftpack.dct(fftpack.dct(dr.T/w).T/h)

sb = np.sign(dcb[dh:, dw:])
sg = np.sign(dcb[dh:, dw:])
sr = np.sign(dcb[dh:, dw:])

sb[sb==0] = 1
sg[sg==0] = 1
sr[sr==0] = 1

allh = np.empty((3, h-dh, w-dw,), dtype=dcb.dtype)
allh[0] = dcb[dh:, dw:]
allh[1] = dcg[dh:, dw:]
allh[2] = dcr[dh:, dw:]
allmin = np.min(np.abs(allh), axis=0)

dcb[dh:, dw:] = sb * allmin 
dcg[dh:, dw:] = sg * allmin 
dcr[dh:, dw:] = sr * allmin 

result = np.empty(shape=datal.shape, dtype=datal.dtype)

result[:,:,0] = fftpack.idct(fftpack.idct(dcb.T).T)
result[:,:,1] = fftpack.idct(fftpack.idct(dcg.T).T)
result[:,:,2] = fftpack.idct(fftpack.idct(dcr.T).T)

result[result<0] = 0
result /= result.max()
resint = np.round(result*255).astype(np.uint8)

#exit()
filtered = Image.fromarray(resint[:,:,::-1])
filtered.save("out.png")
filtered.show()

def callback(event):
    datalcp = datal.copy()
    draw_rad(datalcp, event.y, event.x, rads, gminiscl.get())
    Image.fromarray(np.round(datalcp[:,:,::-1]*255).astype(np.uint8)).show()
    print("clicked at: ", event.x, event.y)

canvas.bind("<Button-1>", callback)
tk.mainloop()
