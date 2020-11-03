
import math
from os.path import join as pjoin
import numpy as np
import vtk
from vtk.util import numpy_support
from fury import actor, window,ui, colormap as cmap
import fury.primitive as fp
from fury.utils import (get_actor_from_polydata, numpy_to_vtk_colors,
                        set_polydata_triangles, set_polydata_vertices,
                        set_polydata_colors,colors_from_actor,
                        vertices_from_actor,update_actor)



vertices_count = 50000
view_size = 300

positions = view_size * \
    np.random.random((vertices_count, 3)) - view_size / 2.0

positions = np.ascontiguousarray(positions,dtype=np.float32);

colors = np.array(cmap.cm.inferno(np.arange(0,vertices_count)/(vertices_count-1)))

radii = 1 + np.random.rand(len(positions))


n_points = colors.shape[0]
np.random.seed(42)
centers = positions # np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# radius = radii * 0.5, #  [1, 1, 2]

scene = window.Scene()

camera = scene.camera()

fs_dec = \
    """
    uniform mat4 MCDCMatrix;
    uniform mat4 MCVCMatrix;


    float sdRoundBox( vec3 p, vec3 b, float r )
    {
        vec3 q = abs(p) - b;
        return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
    }

    float sdEllipsoid( vec3 p, vec3 r )
    {
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
    }
    float sdCylinder(vec3 p, float h, float r)
    {
        vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
        return min(max(d.x,d.y),0.0) + length(max(d,0.0));
    }
    float sdSphere(vec3 pos, float r)
    {
        float d = length(pos) - r;

        return d;
    }
    float map( in vec3 pos)
    {
        float d = sdSphere(pos-0.5, .2);
        float d1 = sdCylinder(pos+0.5, 0.05, .5);
        float d2 = sdEllipsoid(pos + vec3(-0.5,0.5,0), vec3(0.2,0.3,0.5));
        float d3 = sdRoundBox(pos + vec3(0.5,-0.5,0), vec3(0.2,0.1,0.3), .05);


        //.xy

        return min(min(min(d, d1), d2), d3);
    }

    vec3 calcNormal( in vec3 pos )
    {
        vec2 e = vec2(0.0001,0.0);
        return normalize( vec3(map(pos + e.xyy) - map(pos - e.xyy ),
                                map(pos + e.yxy) - map(pos - e.yxy),
                                map(pos + e.yyx) - map(pos - e.yyx)
                                )
                        );
    }

    float castRay(in vec3 ro, vec3 rd)
    {
        float t = 0.0;
        for(int i=0; i < 100; i++)
        {
            vec3 pos = ro + t * rd;
            vec3 nor = calcNormal(pos);

            float h = map(pos);
            if (h < 0.001) break;

            t += h;
            if (t > 20.0) break;
        }
        return t;
    }
    """

fake_sphere = \
"""

vec3 uu = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]); // camera right
vec3 vv = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]); //  camera up
vec3 ww = vec3(MCVCMatrix[0][2], MCVCMatrix[1][2], MCVCMatrix[2][2]); // camera direction
vec3 ro = MCVCMatrix[3].xyz * mat3(MCVCMatrix);  // camera position

// create view ray
vec3 rd = normalize( point.x*-uu + point.y*-vv + ww);
vec3 col = vec3(0.0);

float len = length(point);
float radius = 1.;
if(len > radius)
    {discard;}

//err, lightColor0 vertexColorVSOutput normalVCVSOutput, ambientIntensity; diffuseIntensity;specularIntensity;specularColorUniform;
// float c = len;
// fragOutput0 =  vec4(c,c,c, 1);


vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
vec3 direction = normalize(vec3(1., 1., 1.));
float ddf = max(0, dot(direction, normalizedPoint));
float ssf = pow(ddf, 24);
fragOutput0 = vec4(max(ddf * color, ssf * vec3(1)), 1);
"""


billboard_actor = actor.billboard(centers,
                                    colors=colors,
                                    scales=1.0,
                                    fs_dec=fs_dec,
                                    fs_impl=fake_sphere
                                    )



scene.add(billboard_actor)

showm = window.ShowManager(scene, reset_camera=False, size=(
    1200, 1100), order_transparent=False, multi_samples=2,)


showm.initialize()


scene.set_camera(position=(0, 0, -750))


showm.start()
