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
from fury import actor, window,ui, colormap as cmap

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


from collections import Counter
import networkx as nx
vertices_count = 5000
view_size = 100

# filename = "/Users/filipi/Dropbox/Projects/CDT-Visualization/Networks/Content/WS_10000_10_001-major.xnet"
# filename = "/Users/filipi/Dropbox/Software/Networks 3D Lite/Networks 3D/networks/Wikipedia.xnet"
# filename = "/Users/filipi/Downloads/wax_5000_6_0.xnet"
# network = xnetwork.xnet2igraph("/Volumes/GoogleDrive/My Drive/Remote/Archived/DesktopOLD/Se eu tivesse mesa limpa/Networks/WS_2864_6.xnet")
filename = "/Users/filipi/Dropbox/Projects/FieldsAndAuthorship/Results/Networks/wosAPS.xnet"
network = xnetwork.xnet2igraph(filename)
print(network.vcount())
print(network.vertex_attributes())
maxRank = 10
propertyName = "Cluster Name"
colorValues = {key:index for index,(key,size) in enumerate(Counter(network.vs[propertyName]).most_common())}
colormap = cmap.cm.get_cmap('tab10', maxRank)
clusterRank = np.array([colorValues[value] for value in network.vs[propertyName]])
mask = (np.array(network.vs["Times Cited"])<10)+(clusterRank>=maxRank) + (np.array(network.degree())<=2)
network.delete_vertices(np.where(mask)[0])
network = network.clusters().giant()
okEdges = np.random.choice(network.ecount(),size=int(0.2*network.ecount()),replace=False)
network.delete_edges(okEdges)
network = network.clusters().giant()
edges = np.ascontiguousarray(network.get_edgelist(),dtype=np.uint64)

colorProperty = [colormap(colorValues[value]) if colorValues[value]<maxRank else (0.4,0.4,0.4,1.0) for value in network.vs[propertyName]]

# print(edges.shape)
# edges = np.ascontiguousarray(edges[okEdges,:],dtype=np.uint64)

# print(edges.shape)
vertices_count = network.vcount()
print(vertices_count)
print("Processing...")
# positions = view_size * \
#     np.random.random((vertices_count, 3)) - view_size / 2.0

positions = np.array(network.vs["Position"])
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
degrees = np.array(network.degree())
minDegree = np.min(degrees)
maxDegree = np.max(degrees)
coeff = (degrees-minDegree)/(maxDegree-minDegree)
colors = np.array(cmap.cm.inferno(np.arange(0,vertices_count)/(vertices_count-1)))
colors = np.array(colorProperty)
# (degrees/np.max(degrees)))
###############################################################################
# We define our node size
citations = np.log(np.array(network.vs["Times Cited"]))
radii = 1 + np.power(citations/np.max(citations),2)*10 #np.random.rand(len(positions))

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
                            theta=10,
                            phi=10)


sphere_geometry = numpy_support.vtk_to_numpy(sphere_actor.GetMapper().GetInput().GetPoints().GetData())
geometry_length = sphere_geometry.shape[0] / positions.shape[0]

sphere_actor.GetProperty().SetAmbient(0.9);
sphere_actor.GetProperty().SetDiffuse(0.2);
sphere_actor.PickableOff()

# global picker

# picker = window.vtk.vtkWorldPointPicker()
# picker = window.vtk.vtkPropPicker()
# scenePicker = vtk.vtkScenePicker()


# def left_click_callback(obj, event):

#     global text_block, showm, picker,scenePicker
#     x, y, z = obj.GetCenter()
#     event_pos = showm.iren.GetEventPosition()

#     # picker.Pick(event_pos[0], event_pos[1],
#     #             0, showm.scene)
#     # print(picker.GetPickPosition())
#     cell_index = scenePicker.GetCellId(event_pos)
#     point_index = scenePicker.GetVertexId(event_pos)
#     text = 'Object ID '+str(int(np.floor(point_index/geometry_length)))+"\n"+'Face ID ' + str(cell_index) + '\n' + 'Point ID ' + str(point_index)
#     # # text_block.message = text
#     print(text)
#     # print(cell_index)


# sphere_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1)


# scenePicker.SetEnableVertexPicking(True)

lines_actor = actor.line(np.zeros((len(edges), 2, 3)),
                        colors=edges_colors, lod=False,
                        fake_tube=False,depth_cue=False, linewidth=3,
                        opacity=0.15)

lines_actor.GetProperty().SetAmbient(0.9);
lines_actor.GetProperty().SetDiffuse(0.2);
lines_actor.PickableOff()
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

    # threadID = helios.startAsyncLayout(edgesArray,positions,velocities);
    def iterateHelios(iterationCount):
        for i in range(iterationCount):
            helios.layout(edgesArray,positions,velocities,0.001,0.1,0.02);
        # t1 = threading.Thread(target=heliosLayout, args=(iterationCount,edgesArray,pos,velocities)) 
        # t1.start()
        # t1.join()

    

    def _timer(_obj, _event):
        nonlocal counter
        counter += 1
        # if(counter%100==0):
        #     print(scene.frame_rate())
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



scene = window.Scene()

scene.background((0.0, 0.0, 0.0))

basic_passes = vtk.vtkRenderStepsPass()
ssao = vtk.vtkSSAOPass()
sceneSize = 2000
ssao.SetRadius(0.1 * sceneSize);
ssao.SetBias(0.001 * sceneSize);
ssao.SetKernelSize(512);
# ssao.BlurOff();

ssao.SetDelegatePass(basic_passes)
glrenderer = vtk.vtkOpenGLRenderer.SafeDownCast(scene)
glrenderer.SetPass(ssao)



global text_block
text_block = ui.TextBlock2D(font_size=20, bold=True)
text_block.message = ''
text_block.color = (1, 1, 1)

camera = scene.camera()

scene.add(lines_actor)
scene.add(sphere_actor)


###############################################################################
# The final step ! Visualize the result of our creation! Also, we need to move
# the camera a little bit farther from the network. you can increase the
# parameter max_iteractions of the timer callback to let the animation run for
# more time.

showm = window.ShowManager(scene, reset_camera=False, size=(
    1096, 1096), order_transparent=False, multi_samples=2,)


showm.initialize()

# scenePicker.SetRenderer(showm.scene)

scene.set_camera(position=(0, 0, -1750))

timer_callback = new_layout_timer(
    showm, edges, vertices_count,
    max_iterations=2000,
    vertex_initial_positions=positions)


# Run every 16 milliseconds
try:
    showm.add_timer_callback(False, 16, timer_callback)

    showm.start()

    # window.record(showm.scene, size=(900, 768),
    #             out_path="viz_animated_networks.png")

except KeyboardInterrupt:
    helios.stopAsyncLayout(threadID)
    threadID=0
except Exception as e:
    helios.stopAsyncLayout(threadID)
    threadID=0
    raise

helios.stopAsyncLayout(threadID)