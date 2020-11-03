
import math
from os.path import join as pjoin
import numpy as np
import vtk
from vtk.util import numpy_support
from fury import actor, window,ui, colormap as cmap
import fury.primitive as fp
from fury.utils import (get_actor_from_polydata, numpy_to_vtk_colors,
                        set_polydata_triangles, set_polydata_vertices,
                        set_polydata_colors)

linesCount = 5000
view_size = 300

lines_actor = actor.line(view_size*(np.random.random((linesCount, 2, 3))-0.5),
                        colors=np.random.random((linesCount, 3)), lod=False,
                        fake_tube=True, linewidth=6,
                        opacity=0.8
                        )

scene = window.Scene()
scene.background((0.0, 0.0, 0.0))
camera = scene.camera()

scene.add(lines_actor)

showm = window.ShowManager(scene, reset_camera=False, size=(
    1980, 1920), order_transparent=False, multi_samples=2,)

showm.initialize()
scene.set_camera(position=(0, 0, -750))
showm.start()
