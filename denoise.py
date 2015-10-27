import numpy as np
import pyopencl as cl
from math import sin, cos, pi
from mako.template import Template


tplsrc = """

static const ${dtype} c0 = 1. / sqrt(2.) * sqrt(2. / 8.);
static const ${dtype} c1 = cos(M_PI * 1. / 16.) * sqrt(2. / 8.);
static const ${dtype} c2 = cos(M_PI * 2. / 16.) * sqrt(2. / 8.);
static const ${dtype} c3 = cos(M_PI * 3. / 16.) * sqrt(2. / 8.);
static const ${dtype} c4 = cos(M_PI * 4. / 16.) * sqrt(2. / 8.);
static const ${dtype} c5 = cos(M_PI * 5. / 16.) * sqrt(2. / 8.);
static const ${dtype} c6 = cos(M_PI * 6. / 16.) * sqrt(2. / 8.);
static const ${dtype} c7 = cos(M_PI * 7. / 16.) * sqrt(2. / 8.);

#define a x[0]
// etc

void dct_ii_8a(const ${dtype} x[8], ${dtype} X[8]) {
  X[0] = a*c0 + b*c0 + c*c0 + d*c0 + e*c0 + f*c0 + g*c0 + h*c0;
  X[1] = a*c1 + b*c3 + c*c5 + d*c7 - e*c7 - f*c5 - g*c3 - h*c1;
  X[2] = a*c2 + b*c6 - c*c6 - d*c2 - e*c2 - f*c6 + g*c6 + h*c2;
  X[3] = a*c3 - b*c7 - c*c1 - d*c5 + e*c5 + f*c1 + g*c7 - h*c3;
  X[4] = a*c4 - b*c4 - c*c4 + d*c4 + e*c4 - f*c4 - g*c4 + h*c4;
  X[5] = a*c5 - b*c1 + c*c7 + d*c3 - e*c3 - f*c7 + g*c1 - h*c5;
  X[6] = a*c6 - b*c2 + c*c2 - d*c6 - e*c6 + f*c2 - g*c2 + h*c6;
  X[7] = a*c7 - b*c5 + c*c3 - d*c1 + e*c1 - f*c3 + g*c5 - h*c7;
}

__kernel void filter(__global uchar *gdatain, __global uchar *gdataout){
    size_t gy = get_global_id(0);
    size_t gx = get_global_id(1);
    size_t idx = gy*${w} + gx;
    size_t x, y;
    ${dtype} data[8];
    ${dtype} dcmin = INFINITY;
    ${dtype} dccurrent;

    char rads[${len(rads)}][${len(rads[0])}][2] = {${(',\\\\\\n'+28*' ').join(['{'+', '.join(['{'+str(a)+', '+str(b)+'}' for a, b in coords])+'}' for coords in rads])}};
    for(uint i=0; i<${len(rads)}; i++){
        for(uint j=0; j<${len(rads[0])}; j++){
            y = rads[i][j][0];
            x = rads[i][j][1];
            data[j] = (${dtype}) 1.0*gdatain[idx + ${w}*y + x];
        }
    }

}
"""

tpl = Template(tplsrc)


def get_radius(r, angle):
    res = []
    for i in range(r):
        x = cos(2*pi*angle/360)*i
        y = sin(2*pi*angle/360)*i
        res.append((round(y), round(x)))
    return res

rads = []
for angle in range(0, 360, 5):
    rads.append(get_radius(8, angle))
print(tpl.render(rads=rads, w=512, dtype='float'))



