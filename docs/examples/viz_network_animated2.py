"""
=======================================================
Visualize Networks (Animated version)
=======================================================

The goal of this demo is to show how to visualize a
complex network and use an force directed algorithm to
layout the network. A simpler animation of the network
made by adding some random displacements to nodes
positions is also demoed.

"""

###############################################################################
# First, let's import some useful functions

import math
from os.path import join as pjoin
import numpy as np
import vtk
from vtk.util import numpy_support
from fury import actor, window, colormap as cmap

import helios
import xnetwork
import igraph as ig
import numpy as np

###############################################################################
# This demo has two modes.
# Use `mode = 0` to visualize a randomly generated geographic network by
# iterating it using a force-directed layout heuristic.
#
# Use `mode = 1` to visualize a large network being animated with random
# displacements
#


###############################################################################
# Then let's download some available datasets. (mode 1)

# from fury.data.fetcher import fetch_viz_wiki_nw

# files, folder = fetch_viz_wiki_nw()
# categories_file, edges_file, positions_file = sorted(files.keys())

###############################################################################
# We read our datasets (mode 1)

    # positions = np.loadtxt(pjoin(folder, positions_file))
    # categories = np.loadtxt(pjoin(folder, categories_file), dtype=str)
    # edges = np.loadtxt(pjoin(folder, edges_file), dtype=np.int)
    # vertices_count = len(positions)

###############################################################################
# Generate a geographic random network, requires networkx package (mode 0)


import networkx as nx
vertices_count = 5000
view_size = 100

filename = "/Users/filipi/Dropbox/Projects/CDT-Visualization/Networks/Content/WS_10000_10_001-major.xnet"
# filename = "/Users/filipi/Dropbox/Software/Networks 3D Lite/Networks 3D/networks/Wikipedia.xnet"
# filename = "/Users/filipi/Downloads/wax_5000_6_0.xnet"
# network = xnetwork.xnet2igraph("/Volumes/GoogleDrive/My Drive/Remote/Archived/DesktopOLD/Se eu tivesse mesa limpa/Networks/WS_2864_6.xnet")

network = xnetwork.xnet2igraph(filename)
# network = xnetwork.xnet2igraph("/Volumes/GoogleDrive/My Drive/Dropbox/Motifs/Orbits/networks/WAX5000-2D-main.xnet")

edges = np.ascontiguousarray(network.get_edgelist(),dtype=np.uint64)
vertices_count = network.vcount()

print("Processing...")
positions = view_size * \
    np.random.random((vertices_count, 3)) - view_size / 2.0

positions = np.ascontiguousarray(positions,dtype=np.float32);

###############################################################################
# We attribute a color to each category of our dataset which correspond to our
# nodes colors.

# categories = np.arange(0, vertices_count)
# category2index = {category: i
#                 for i, category in enumerate(np.unique(categories))}

# index2category = np.unique(categories)

# category_colors = cmap.distinguishable_colormap(nb_colors=len(index2category))

# colors = np.array([category_colors[category2index[category]]
#                 for category in categories])
degrees = network.degree();
colors = np.array(cmap.cm.inferno(np.arange(0,vertices_count)/(vertices_count-1)))
# (degrees/np.max(degrees)))
###############################################################################
# We define our node size

radii = 1 + np.random.rand(len(positions))

###############################################################################
# Lets create our edges now. They will indicate a citation between two nodes.
# The colors of each edge are interpolated between the two endpoints.

edges_colors = []
for source, target in edges:
    edges_colors.append(np.array([colors[source], colors[target]]))

edges_colors = np.average(np.array(edges_colors), axis=1)


###############################################################################
# Our data preparation is ready, it is time to visualize them all. We start to
# build 2 actors that we represent our data : sphere_actor for the nodes and
# lines_actor for the edges.

sphere_actor = actor.sphere(centers=np.zeros(positions.shape),
                            colors=colors,
                            radii=radii * 0.5,
                            theta=8,
                            phi=8)


lines_actor = actor.line(np.zeros((len(edges), 2, 3)),
                        colors=edges_colors, lod=False,
                        fake_tube=True, linewidth=3,opacity=0.2)

###############################################################################
# Defining timer callback and layout iterator




velocities = np.zeros((vertices_count, 3),dtype=np.float32)
edgesArray = np.ascontiguousarray(np.array(edges),dtype=np.uint64);
threadID = 0;
def new_layout_timer(showm, edges_list, vertices_count,
                    max_iterations=1000, vertex_initial_positions=None):
    global threadID
    counter = 0
    view_size = 500
    viscosity = 0.10
    alpha = 0.5
    a = 0.0005
    b = 1.0
    deltaT = 1.0
    if(vertex_initial_positions is not None):
        positions[:] = np.array(vertex_initial_positions)
    else:
        positions[:] = view_size * \
            np.random.random((vertices_count, 3)) - view_size / 2.0

    sphere_geometry = np.array(numpy_support.vtk_to_numpy(
        sphere_actor.GetMapper().GetInput().GetPoints().GetData()))
    geometry_length = sphere_geometry.shape[0] / vertices_count

    threadID = helios.startAsyncLayout(edgesArray,positions,velocities);
    def iterateHelios(iterationCount):
        for i in range(iterationCount):
            helios.layout(edgesArray,positions,velocities,0.001,0.1,0.02);
        # t1 = threading.Thread(target=heliosLayout, args=(iterationCount,edgesArray,pos,velocities)) 
        # t1.start()
        # t1.join()

    

    def _timer(_obj, _event):
        nonlocal counter
        counter += 1

        spheres_positions = numpy_support.vtk_to_numpy(
            sphere_actor.GetMapper().GetInput().GetPoints().GetData())
        spheres_positions[:] = sphere_geometry + \
            np.repeat(positions, geometry_length, axis=0)

        edges_positions = numpy_support.vtk_to_numpy(
            lines_actor.GetMapper().GetInput().GetPoints().GetData())
        edges_positions[::2] = positions[edges_list[:, 0]]
        edges_positions[1::2] = positions[edges_list[:, 1]]

        lines_actor.GetMapper().GetInput().GetPoints().GetData().Modified()
        lines_actor.GetMapper().GetInput().ComputeBounds()

        sphere_actor.GetMapper().GetInput().GetPoints().GetData().Modified()
        sphere_actor.GetMapper().GetInput().ComputeBounds()
        showm.scene.ResetCameraClippingRange()
        showm.scene.azimuth(0.05)
        # showm.scene.yaw(0.05)
        showm.render()

        if counter >= max_iterations:
            showm.exit()
    return _timer

###############################################################################
# All actors need to be added in a scene, so we build one and add our
# lines_actor and sphere_actor.

window.frame_rate
scene = window.Scene()

camera = scene.camera()

# scene.add(lines_actor)
scene.add(sphere_actor)

###############################################################################
# The final step ! Visualize the result of our creation! Also, we need to move
# the camera a little bit farther from the network. you can increase the
# parameter max_iteractions of the timer callback to let the animation run for
# more time.

showm = window.ShowManager(scene, reset_camera=False, size=(
    1980, 1920), order_transparent=False, multi_samples=8,)

showm.initialize()

scene.set_camera(position=(0, 0, -750))

timer_callback = new_layout_timer(
    showm, edges, vertices_count,
    max_iterations=2000,
    vertex_initial_positions=positions)


# Run every 16 milliseconds
try:
    showm.add_timer_callback(True, 16, timer_callback)

    showm.start()

    window.record(showm.scene, size=(900, 768),
                out_path="viz_animated_networks.png")

except KeyboardInterrupt:
    helios.stopAsyncLayout(threadID)
    threadID=0
except Exception as e:
    helios.stopAsyncLayout(threadID)
    threadID=0
    raise

helios.stopAsyncLayout(threadID)