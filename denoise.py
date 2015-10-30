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

#define c0 ${Decimal(1. / sqrt(2.) * sqrt(2. / radius))}
% for i in range(1,radius+2):
#define c${i} ${Decimal(cos(pi * i / (2*radius)) * sqrt(2. / radius))}
% endfor


${dtype+s} fstval(const ${dtype+s} x[${radius}]){
    return ${' '.join(['{0}x[{2}]*({3})c{1}'.format(c[0], c[1], j, dtype) for j,c in enumerate(allc[n])])};
}


${dtype+s} highfq(const ${dtype+s} x[${radius}]) {
    return ${' '.join(['{0}x[{2}]*({3})c{1}'.format(c[0], c[1], j, dtype) for j,c in enumerate(allct[radius-1])])};
}

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
    ${dtype} dcmin = INFINITY;
    ${dtype} dcsum;
    ${dtype+s} dccurrent;
    
    const char allcrds[${len(crds)}][2] = {${','.join(['{'+str(a)+', '+str(b)+'}' for a, b in crds])}};
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
    const char rads[${len(rads)}][${len(rads[0])}] = {${(',\\\\\\n'+33*' ').join(['{'+', '.join([str(a) for a in coords])+'}' for coords in rads])}};
    for(uint i=0; i<${rads_cnt}; i++){
        for(uint j=0; j<${radius}; j++){
            c = rads[i][j];
            data[j] = (${dtype+s}) alldata[c];
        }
% if numc>1:
        dccurrent = highfq(data);
        dcsum = ${'+'.join(['dccurrent.s'+str(i) for i in range(numc)])};
% else:
        dcsum = highfq(data);
% endif
        mini  = select(mini, i, (uint)(dcsum<dcmin));
        dcmin = select(dcmin, dcsum, (int)(mini==i));
    }
    for(uint j=0; j<${radius}; j++){
        c = rads[mini][j];
        data[j] = (${dtype+s}) alldata[c];
    }
    gminis[idx] = mini;
    dct_ii_${radius}a(data, dctf);
% for i in range(3, radius):
    dctf[${i}] /= ${i};
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
rr = 8
nn = 2
denom = 2*rr

def arr_from_np(queue, nparr):
    if nparr.dtype == np.object:
        nparr = np.concatenate(nparr)
    buf = cl.Buffer(ctx, mf.READ_WRITE| mf.COPY_HOST_PTR, hostbuf=nparr)
    return clarray.Array(queue, nparr.shape, nparr.dtype, data=buf)

allcircle = []
def get_radius(n, r, angle):
    res = []
    for i in range(-n, r-n):
        x = round( cos(2*pi*angle/360)*i )
        y = round( sin(2*pi*angle/360)*i )
        if not [x,y] in allcircle:
            allcircle.append([x,y])
        res.append(allcircle.index([x,y]))
    return res

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
allct = np.array(allc).transpose(1,0,2).tolist()

rads = []
for angle in range(0, 360, 5):
    rads.append(get_radius(nn, rr, angle))

datal = cv2.imread("../cvrecogn/cicada_molt_stereo_pair_by_markdow.jpg")/255

datal *= 0.9
datal += np.random.rand(datal.size).reshape(datal.shape)*0.1

datalcl = arr_from_np(queue, datal.astype(np.float32))



res = clarray.zeros_like(datalcl)
gminiscl = clarray.zeros_like(datalcl).astype(np.uint32)

h, w = datalcl.shape[:2]

ksource = tpl.render(rads=rads, w=w, allc=allc, allct=allct, n=nn, dtype='float', crds=allcircle, numc=3)
print(ksource)
#exit()

program = cl.Program(ctx, ksource).build()
program.filter(queue, (h-2*rr, w-2*rr,), None, datalcl.ravel().data, res.data, gminiscl.data)

resint = np.round(res.get()*255).astype(np.uint8)
from PIL import Image
Image.fromarray(np.round(datalcl.get()[:,:,::-1]*255).astype(np.uint8)).show()
Image.fromarray(resint[:,:,::-1]).show()
