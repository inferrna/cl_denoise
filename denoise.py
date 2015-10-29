import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
from math import sin, cos, pi
from mako.template import Template
import cv2

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


tplsrc = """
#ifndef INFINITY
#define INFINITY 1.0/0
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define a x[0]
#define b x[1]
#define c x[2]
#define d x[3]
#define e x[4]
#define f x[5]
#define g x[6]
#define h x[7]
<% radius = len(rads[0]) %>
<% rads_cnt = len(rads) %>

${dtype} fstval(const ${dtype} x[8]){
    static const ${dtype} c0 = 1. / sqrt((${dtype})2.) * sqrt((${dtype})(2. / ${radius}.));
% for i in range(1,radius):
    static const ${dtype} c${i} = cos((${dtype})(M_PI * ${i}. / ${2*radius}.)) * sqrt((${dtype})(2. / ${radius}.));
% endfor
    return a*c0 + b*c1 + c*c2 + d*c3 + e*c4 + f*c5 + g*c6 + h*c7;
    /*X[1] = a*c0 + b*c3 + c*c6 - d*c7 - e*c4 - f*c1 - g*c2 - h*c5;
    X[2] = a*c0 + b*c5 - c*c6 - d*c1 - e*c4 + f*c7 + g*c2 + h*c3;
    X[3] = a*c0 + b*c7 - c*c2 - d*c5 + e*c4 + f*c3 - g*c6 - h*c1;
    X[4] = a*c0 - b*c7 - c*c2 + d*c5 + e*c4 - f*c3 - g*c6 + h*c1;
    X[5] = a*c0 - b*c5 - c*c6 + d*c1 - e*c4 - f*c7 + g*c2 - h*c3;
    X[6] = a*c0 - b*c3 + c*c6 + d*c7 - e*c4 + f*c1 - g*c2 + h*c5;
    X[7] = a*c0 - b*c1 + c*c2 - d*c3 + e*c4 - f*c5 + g*c6 - h*c7;*/
}


${dtype} highfq(const ${dtype} x[8]) {
    static const ${dtype} c0 = 1. / sqrt((${dtype})2.) * sqrt((${dtype})(2. / ${radius}.));
% for i in range(1,radius):
    static const ${dtype} c${i} = cos((${dtype})(M_PI * ${i}. / ${2*radius}.)) * sqrt((${dtype})(2. / ${radius}.));
% endfor
    return a*c7 - b*c5 + c*c3 - d*c1 + e*c1 - f*c3 + g*c5 - h*c7;
}

void dct_ii_8a(const ${dtype} x[8], ${dtype} X[8]) {
    static const ${dtype} c0 = 1. / sqrt((${dtype})2.) * sqrt((${dtype})(2. / ${radius}.));
% for i in range(1,radius):
    static const ${dtype} c${i} = cos((${dtype})(M_PI * ${i}. / ${2*radius}.)) * sqrt((${dtype})(2. / ${radius}.));
% endfor
% for i in range(0,radius):
    X[${i}] = ${' '.join(['{1}*x[{0}]'.format(c, j) for c,j in enumerate(allct[i])])};
% endfor
/*
    X[0] = a*c0 + b*c0 + c*c0 + d*c0 + e*c0 + f*c0 + g*c0 + h*c0;
    X[1] = a*c1 + b*c3 + c*c5 + d*c7 - e*c7 - f*c5 - g*c3 - h*c1;
    X[2] = a*c2 + b*c6 - c*c6 - d*c2 - e*c2 - f*c6 + g*c6 + h*c2;
    X[3] = a*c3 - b*c7 - c*c1 - d*c5 + e*c5 + f*c1 + g*c7 - h*c3;
    X[4] = a*c4 - b*c4 - c*c4 + d*c4 + e*c4 - f*c4 - g*c4 + h*c4;
    X[5] = a*c5 - b*c1 + c*c7 + d*c3 - e*c3 - f*c7 + g*c1 - h*c5;
    X[6] = a*c6 - b*c2 + c*c2 - d*c6 - e*c6 + f*c2 - g*c2 + h*c6;
    X[7] = a*c7 - b*c5 + c*c3 - d*c1 + e*c1 - f*c3 + g*c5 - h*c7;
*/
}

__kernel void filter(__global ${dtype} *gdatain, __global ${dtype} *gdataout, __global uint *gminis){
    size_t gy = get_global_id(0) + ${radius}; //With offset
    size_t gx = get_global_id(1) + ${radius}; //With offset
    size_t idx = gy*${w} + gx;
    size_t x, y, mini = 0;
    ${dtype} data[8];
    ${dtype} dctf[8];
    ${dtype} dcmin = INFINITY;
    ${dtype} dccurrent;

    static const char rads[${len(rads)}][${len(rads[0])}][2] = {${(',\\\\\\n'+40*' ').join(['{'+', '.join(['{'+str(a)+', '+str(b)+'}' for a, b in coords])+'}' for coords in rads])}};
    for(uint i=0; i<${rads_cnt}; i++){
        for(uint j=0; j<${radius}; j++){
            y = rads[i][j][0];
            x = rads[i][j][1];
            data[j] = (${dtype}) 1.0*gdatain[idx + ${w}*y + x];
        }
        dccurrent = highfq(data);
        mini  = select(mini, i, (int)(dccurrent<dcmin));
        dcmin = select(dcmin, dccurrent, (uint)(mini==i));
    }
    for(uint j=0; j<${radius}; j++){
        y = rads[mini][j][0];
        x = rads[mini][j][1];
        data[j] = (${dtype}) 1.0*gdatain[idx + ${w}*y + x];
    }
    gminis[idx] = mini;
    dct_ii_8a(data, dctf);
    dctf[7] /= 4;
    dctf[6] /= 3.5;
    dctf[5] /= 3;
    dctf[4] /= 2.5;
    dctf[3] /= 2;
    dctf[2] /= 1.5;
    gdataout[idx] = clamp((${dtype})fstval(dctf), (${dtype})0.0, (${dtype})1.0);
}
"""

tpl = Template(tplsrc)

def arr_from_np(queue, nparr):
    if nparr.dtype == np.object:
        nparr = np.concatenate(nparr)
    buf = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return clarray.Array(queue, nparr.shape, nparr.dtype, data=buf)


def get_radius(r, angle):
    res = []
    for i in range(r):
        x = cos(2*pi*angle/360)*i
        y = sin(2*pi*angle/360)*i
        res.append((round(y), round(x)))
    return res

rr = 8
denom = 2*rr
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
    line.append("{0}c{1}".format(pre, num))
  allc.append(line)
import numpy as np
allct = np.array(allc).T.tolist()

rads = []
for angle in range(0, 360, 5):
    rads.append(get_radius(rr, angle))

datal = cv2.imread("../cvrecogn/cicada_molt_stereo_pair_by_markdow.jpg")[:,:,0]/255

datal *= 0.9
datal += np.random.rand(datal.size).reshape(datal.shape)*0.1

datalcl = arr_from_np(queue, datal.astype(np.float32))



res = clarray.zeros_like(datalcl)
gminiscl = clarray.zeros_like(datalcl).astype(np.uint32)

h, w = datalcl.shape

ksource = tpl.render(rads=rads, w=w, allc=allc, allct=allct, dtype='float')
print(ksource)


program = cl.Program(ctx, ksource).build()
program.filter(queue, (h-2*rr, w-2*rr,), None, datalcl.ravel().data, res.data, gminiscl.data)

resint = np.round(res.get()*255).astype(np.uint8)
from PIL import Image
Image.fromarray(np.round(datalcl.get()*255).astype(np.uint8)).show()
Image.fromarray(resint).show()
