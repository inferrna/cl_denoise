import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
from math import sin, cos, pi
from mako.template import Template
from scipy import fftpack
import cv2
import sys

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

tplsrc = """
<% from math import cos,sqrt %>
<% from decimal import Decimal %>
<% radius = len(rads[0]) %>
<% rads_cnt = len(rads) %>
<% pi = 3.14159265358979323846 %>
<% s = str(numc) if numc>1 else ''  %>
#ifndef INFINITY
#define INFINITY 1.0/0
#endif
#ifndef M_PI
#define M_PI ${Decimal(pi)}
#endif

#define amp2(a) (max(a.s0, a.s1) - min(a.s0, a.s1))
#define amp3(a) (max(max(a.s0, a.s1), a.s2) - min(min(a.s0, a.s1), a.s2))
#define amp4(a) (max(max(a.s0, a.s1), max(a.s2, a.s3)) - min(min(a.s0, a.s1), min(a.s2, a.s3)))

#define c0 ${Decimal(1. / sqrt(2.) * sqrt(2. / radius))}
//Generated DCT coefficients
% for i in range(1,radius+2):
#define c${i} ${Decimal(cos(pi * i / (2*radius)) * sqrt(2. / radius))}
% endfor


//Return value of only n-element from rdct
${dtype+s} fstval(const ${dtype+s} x[${radius}]){
    return ${' '.join(['{0}x[{2}]*({3})c{1}'.format(c[0], c[1], j, dtype) for j,c in enumerate(allc[n])])};
}

//Return value of only last from dct
${dtype+s} highfq(const ${dtype+s} x[${radius}]) {
    //allc[n][r][0]
    return ${'\\n+'.join(['fabs('+''.join(['{0}x[{2}]*({3})c{1}'.format(c[0], c[1], j, dtype) for j,c in enumerate(allct[r])])+')' for r in range(1, radius)])};
    //return ${' '.join(['{0}x[{2}]*({3})c{1}'.format(c[0], c[1], j, dtype) for j,c in enumerate(allct[radius-1])])};
}

//Calculate full dct-${radius}
void dct_ii_${radius}a(const ${dtype+s} x[${radius}], ${dtype+s} X[${radius}]) {
% for i in range(0,radius):
    X[${i}] = ${' '.join(['{0}x[{2}]*({3})c{1}'.format(c[0], c[1], j, dtype) for j,c in enumerate(allct[i])])};
% endfor
}

__kernel void filter(__global ${dtype} *gdatain, __global ${dtype} *gdataout, __global uint *gminis){
    size_t gy = get_global_id(0) + ${radius}; //With offset
    size_t gx = get_global_id(1) + ${radius}; //With offset
    size_t idx = gy*${w} + gx;
    uint x, y, mini, c = 0;
    ${dtype+s} alldata[${len(crds)}];
    ${dtype+s} data[${radius}];
    ${dtype+s} dctf[${radius}];
    ${dtype+s} result;
    ${dtype} dcsmin = INFINITY; //Minimal sum of last fqs.
    ${dtype} dcsum;             //Sum of last fqs
    ${dtype} amplh;             //Difference betw max and min of hfq of diff colors. If small - seems it not an noise.
    ${dtype+s} dccurrent;
    ${dtype+s} dcmin = (${dtype+s})(${', '.join(numc*['INFINITY'])}); //Last fqs corresponding dcsmin
    const char allcrds[${len(crds)}][2] = {${','.join(['{'+str(a)+', '+str(b)+'}' for a, b in crds])}}; // Cache of round pixels
    // Radiuses construct from allcrds pixels
    const char rads[${len(rads)}][${len(rads[0])}] = {${(',\\\\\\n'+33*' ').join(['{'+', '.join([str(a) for a in coords])+'}' for coords in rads])}};

    for(uint i=0; i<${len(crds)}; i++){
        x = allcrds[i][0];
        y = allcrds[i][1];
% if numc>1:
% for i in range(numc):
        alldata[i].s${i} = gdatain[${numc}*(idx + ${w}*y + x)+${i}];
% endfor
% else:
        alldata[i] = gdatain[idx + ${w}*y + x];
% endif
    } 
    //Find best direction
    for(uint i=0; i<${rads_cnt}; i++){
        for(uint j=0; j<${radius}; j++){
            c = rads[i][j];
            data[j] = (${dtype+s}) alldata[c];
        }
% if numc>1:
        //TODO: move signing to highfq function
        dccurrent = highfq(data);
        dcsum = ${'+'.join(['dccurrent.s'+str(i) for i in range(numc)])};
        dcmin = select(dcmin, dccurrent, (uint${s})(dcsum<dcsmin)); //??-1
% else:
        dcsum = highfq(data);
% endif
        mini  = select(mini, i, (uint)(dcsum<dcsmin));
        dcsmin = select(dcsmin, dcsum, (int)(mini==i));
    }
    //Get line of best direction found
    for(uint j=0; j<${radius}; j++){
        c = rads[mini][j];
        data[j] = (${dtype+s}) alldata[c];
    }
    gminis[idx] = mini; //Store minimal index for debug vis
    dct_ii_${radius}a(data, dctf); //Compute 1-d dct-${radius} of best direction
    dcmin = fabs(dctf[${radius-1}]); //??-1
    amplh = fabs(amp${numc}(dctf[${radius-1}]) + amp${numc}(dctf[${radius-2}])); //OR dcmin
    //if(gx==1000) printf("amplh == %f\\n", amplh);
    //printf("Thread %u %u. amplh == %f\\n", gy, gx, amplh);
    //Divide fq to 1 (no divide at all) when: sign*fq < 0 (always positive noise, may be here I'm wrong !!TODO: check it)
    //and/or (??-2) max fq < 0.1 (not an noisy color)
    ${'int'+s} dvdr = (${'int'+s})(${', '.join(numc*['1'])});
    dvdr = select(dvdr, (${'int'+s})(${', '.join(numc*['2'])}), (dcmin>(${dtype})${radius/128}));
    dvdr = select(dvdr, (${'int'+s})(${', '.join(numc*['3'])}), (dcmin>(${dtype})${radius/96}));
    dvdr = select(dvdr, (${'int'+s})(${', '.join(numc*['4'])}), (dcmin>(${dtype})${radius/64}));
    //if(gx==${radius}) printf("dvdr == %d, %d, %d\\n", dvdr.s0, dvdr.s1, dvdr.s2);    
    //if(gx==${radius}) printf("dcmin == %f, %f, %f\\n", dcmin.s0, dcmin.s1, dcmin.s2);    
    //if(gx==${radius}) printf("amplh == %f\\n", amplh);    
% for i in range(radius//3, radius):
% if numc>1:
% for j in range(numc):
    dctf[${i}].s${j} /= select(dvdr.s${j}*${i}, 1, (uint)(amplh<.015));
% endfor
% else:
    dctf[${i}].s${j} /= select(${i}, 1, (dctf[${i}]<0.1); //
% endif    
% endfor
% if numc>1:
    result = clamp((${dtype+s})fstval(dctf), (${dtype+s})(${', '.join(['0.0']*numc)}), (${dtype+s})(${', '.join(['1.0']*numc)}));
% for i in range(numc):
    gdataout[idx*${numc} + ${i}] = result.s${i};
% endfor
% else:
    gdataout[idx] = clamp((${dtype})fstval(dctf), (${dtype})0.0, (${dtype})1.0);
% endif    
}
"""

tpl = Template(tplsrc)
rr = 4
nn = 1
denom = 2*rr

def draw_rad(array, y, x, rads, gminis):
    arr = []
    for i, (dx, dy) in enumerate([allcircle[c] for c in rads[gminiscl[y, x].get()]]):
        arr.append(array[y+dy, x+dx,:].copy())
        array[y+dy, x+dx,:] = 1.0
        if i==nn:
            array[y+dy, x+dx,1] = 0.0
    nparr = np.array(arr)
    dctarr = fftpack.dct(arr, axis=0)
    print(nparr)
    print(dctarr)
#    Image.fromarray(np.round(array[:,:,::-1]*255).astype(np.uint8)).show()


def arr_from_np(queue, nparr):
    if nparr.dtype == np.object:
        nparr = np.concatenate(nparr)
    buf = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return clarray.Array(queue, nparr.shape, nparr.dtype, data=buf)

# Collect pixel coordinates of the round 
allcircle = []
# Calculate x, y coordinates of radial lines.
def get_radius(n, r, angle):
    res = []
    for i in range(-n, r-n):
        x = round( cos(2*pi*angle/360)*i )
        y = round( sin(2*pi*angle/360)*i )
        pc = [x,y]  # Swapped. TODO: recheck it
        if not pc in allcircle:
            allcircle.append(pc)
        res.append(allcircle.index(pc)) # Coordinates mapped to pixels collection
    return res


# Calculate position of rdct coefficients (cx) with sign
allc = []
for n in range(rr):
  line = []
  for k in range(rr):
    num = k * (2*n + 1)
    while num > denom * 2: num -= denom * 2
    if num > denom: num = 2 * denom - num
    if num > denom / 2:
      pre = "-"
      num = denom - num
    else:
        pre = '+'
    line.append((pre, num,))
  allc.append(line)
import numpy as np
# Transpose to get position of dct coefficients (cx) with sign
allct = np.array(allc).transpose(1,0,2).tolist()

rads = []
step = round(360/(2*pi*rr) + 0.5)
for angle in range(0, 360+step, step):
    rads.append(get_radius(nn, rr, angle))

#datal = cv2.resize(cv2.imread("../cvrecogn/cicada_molt_stereo_pair_by_markdow.jpg"), (128, 64), interpolation=cv2.INTER_LANCZOS4)/255
#datal = cv2.imread("../cvrecogn/cicada_molt_stereo_pair_by_markdow.jpg")/255
datal = cv2.imread(sys.argv[1] if len(sys.argv) >=2 else "DSC07693.png")/255
h, w = datal.shape[:2]

#datal += np.random.rand(datal.size).reshape(datal.shape)*(1.0-datal)*0.5

#idxdark = datal<0.5
#idxlight = np.min(datal, axis=2)>0.5
#rnd = np.random.rand(datal[idxlight].size//3)*0.25
#datal[idxlight] *= 0.75
#datal[idxlight] += np.array(3*[rnd]).T#.reshape(-1, datal.shape[-1])
#datal[idxdark] += np.random.rand(datal[idxdark].size)*0.25

datalcl = arr_from_np(queue, datal.astype(np.float32))



res = clarray.zeros_like(datalcl)
gminiscl = clarray.zeros(dtype=np.uint32, shape=datalcl.shape[:2], queue=queue)


ksource = tpl.render(rads=rads, w=w, allc=allc, allct=allct, n=nn, dtype='float', crds=allcircle, numc=3)
print(ksource)
#exit()

program = cl.Program(ctx, ksource).build()
program.filter(queue, (h-2*rr, w-2*rr,), None, datalcl.ravel().data, res.data, gminiscl.data)

resint = np.round(res.get()*255).astype(np.uint8)

import tkinter as tk
from PIL import ImageDraw, Image, ImageTk
import sys

window = tk.Tk(className="bla")
original = Image.fromarray(np.round(datal[:,:,::-1]*255).astype(np.uint8))

canvas = tk.Canvas(window, width=original.size[0], height=original.size[1])
canvas.pack()
image_tk = ImageTk.PhotoImage(original)

canvas.create_image(original.size[0]//2, original.size[1]//2, image=image_tk)
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
