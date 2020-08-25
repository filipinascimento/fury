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
import fury.primitive as fp
from fury.utils import (get_actor_from_polydata, numpy_to_vtk_colors,
                        set_polydata_triangles, set_polydata_vertices,
                        set_polydata_colors)

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
view_size = 10

filename = "/Users/filipi/Dropbox/Projects/CDT-Visualization/Networks/Content/WS_10000_10_001-major.xnet"
# filename = "/Users/filipi/Dropbox/Software/Networks 3D Lite/Networks 3D/networks/Wikipedia.xnet"
# filename = "/Users/filipi/Downloads/wax_5000_6_0.xnet"
# network = xnetwork.xnet2igraph("/Volumes/GoogleDrive/My Drive/Remote/Archived/DesktopOLD/Se eu tivesse mesa limpa/Networks/WS_2864_6.xnet")

network = xnetwork.xnet2igraph(filename)

edges = np.ascontiguousarray(network.get_edgelist(),dtype=np.uint64)
print(len(edges))
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


# n_points = vertices_count
n_points = colors.shape[0]
np.random.seed(42)
centers = np.zeros(positions.shape) # np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# radius = radii * 0.5, #  [1, 1, 2]

radius = np.ones(n_points)

polydata = vtk.vtkPolyData()

verts, faces = fp.prim_square()

big_verts = 5*np.tile(verts, (centers.shape[0], 1))
big_cents = np.repeat(centers, verts.shape[0], axis=0)

big_verts += big_cents

# print(big_verts)

big_scales = np.repeat(radius, verts.shape[0], axis=0)


# print(big_scales)

big_verts *= big_scales[:, np.newaxis]

# print(big_verts)

tris = np.array([[0, 1, 2], [2, 3, 0]], dtype='i8')

big_tris = np.tile(tris, (centers.shape[0], 1))
shifts = np.repeat(np.arange(0, centers.shape[0] * verts.shape[0],
                                verts.shape[0]), tris.shape[0])

big_tris += shifts[:, np.newaxis]

# print(big_tris)

big_cols = np.repeat(colors*255, verts.shape[0], axis=0)

# print(big_cols)

big_centers = np.repeat(centers, verts.shape[0], axis=0)

# print(big_centers)

big_centers *= big_scales[:, np.newaxis]

# print(big_centers)

# set_polydata_vertices(polydata, big_verts)
set_polydata_triangles(polydata, big_tris)
# set_polydata_colors(polydata, big_cols)

vtk_verts = numpy_support.numpy_to_vtk(big_verts, deep=False)
vtk_points = vtk.vtkPoints()
vtk_points.SetData(vtk_verts)
polydata.SetPoints(vtk_points)

vtk_colors = numpy_support.numpy_to_vtk(big_cols, deep=True,array_type=vtk.VTK_UNSIGNED_CHAR)
vtk_colors.SetNumberOfComponents(4)
vtk_colors.SetName("RGBA")
polydata.GetPointData().SetScalars(vtk_colors)

vtk_centers = numpy_support.numpy_to_vtk(big_centers, deep=False)
vtk_centers.SetNumberOfComponents(3)
vtk_centers.SetName("center")
polydata.GetPointData().AddArray(vtk_centers)

sphere_actor = get_actor_from_polydata(polydata)
# sphere_actor = actor.square(centers)
sphere_actor.GetProperty().BackfaceCullingOff()

# scene.add(canvas_actor)

mapper = sphere_actor.GetMapper()

mapper.MapDataArrayToVertexAttribute(
    "center", "center", vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    "//VTK::ValuePass::Dec",
    True,
    """
    //VTK::ValuePass::Dec
    in vec3 center;

    uniform mat4 Ext_mat;

    out vec3 centeredVertexMC;
    out vec3 cameraPosition;
    out vec3 viewUp;

    """,
    False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Vertex,
    "//VTK::ValuePass::Impl",
    True,
    """
    //VTK::ValuePass::Impl
    centeredVertexMC = vertexMC.xyz - center;
    float scalingFactor = 1. / abs(centeredVertexMC.x);
    centeredVertexMC *= scalingFactor;

    vec3 CameraRight_worldspace = vec3(MCVCMatrix[0][0], MCVCMatrix[1][0], MCVCMatrix[2][0]);
    vec3 CameraUp_worldspace = vec3(MCVCMatrix[0][1], MCVCMatrix[1][1], MCVCMatrix[2][1]);

    vec3 vertexPosition_worldspace = center + CameraRight_worldspace * 1 * centeredVertexMC.x + CameraUp_worldspace * 1 * centeredVertexMC.y;
    gl_Position = MCDCMatrix * vec4(vertexPosition_worldspace, 1.);

    """,
    False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    "//VTK::ValuePass::Dec",
    True,
    """
    //VTK::ValuePass::Dec
    in vec3 centeredVertexMC;
    in vec3 cameraPosition;
    in vec3 viewUp;

    uniform vec3 Ext_camPos;
    uniform vec3 Ext_viewUp;
    """,
    False
)

mapper.AddShaderReplacement(
    vtk.vtkShader.Fragment,
    "//VTK::Light::Impl",
    True,
    """
    // Renaming variables passed from the Vertex Shader
    vec3 color = vertexColorVSOutput.rgb;
    vec3 point = centeredVertexMC;
    fragOutput0 = vec4(color, 0.7);
    /*
    // Comparing camera position from vertex shader and python
    float dist = distance(cameraPosition, Ext_camPos);
    if(dist < .0001)
        fragOutput0 = vec4(1, 0, 0, 1);
    else
        fragOutput0 = vec4(0, 1, 0, 1);


    // Comparing view up from vertex shader and python
    float dist = distance(viewUp, Ext_viewUp);
    if(dist < .0001)
        fragOutput0 = vec4(1, 0, 0, 1);
    else
        fragOutput0 = vec4(0, 1, 0, 1);
    */
    float len = length(point);
    // VTK Fake Spheres
    float radius = 1.;
    if(len > radius)
        discard;
    vec3 normalizedPoint = normalize(vec3(point.xy, sqrt(1. - len)));
    vec3 direction = normalize(vec3(1., 1., 1.));
    float df = max(0, dot(direction, normalizedPoint));
    float sf = pow(df, 24);
    fragOutput0 = vec4(max(df * color, sf * vec3(1)), 1);
    """,
    False
)

@vtk.calldata_type(vtk.VTK_OBJECT)
def vtk_shader_callback(caller, event, calldata=None):
    res = scene.size()
    camera = scene.GetActiveCamera()
    cam_pos = camera.GetPosition()
    foc_pnt = camera.GetFocalPoint()
    view_up = camera.GetViewUp()
    # cam_light_mat = camera.GetCameraLightTransformMatrix()
    # comp_proj_mat = camera.GetCompositeProjectionTransformMatrix()
    # exp_proj_mat = camera.GetExplicitProjectionTransformMatrix()
    # eye_mat = camera.GetEyeTransformMatrix()
    # model_mat = camera.GetModelTransformMatrix()
    # model_view_mat = camera.GetModelViewTransformMatrix()
    # proj_mat = camera.GetProjectionTransformMatrix(scene)
    view_mat = camera.GetViewTransformMatrix()
    mat = view_mat
    np.set_printoptions(precision=3, suppress=True)
    np_mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            np_mat[i, j] = mat.GetElement(i, j)
    program = calldata
    if program is not None:
        # print("\nCamera position: {}".format(cam_pos))
        # print("Focal point: {}".format(foc_pnt))
        # print("View up: {}".format(view_up))
        # print(mat)
        # print(np_mat)
        # print(np.dot(-np_mat[:3, 3], np_mat[:3, :3]))
        # a = np.array(cam_pos) - np.array(foc_pnt)
        # print(a / np.linalg.norm(a))
        # print(cam_light_mat)
        # #print(comp_proj_mat)
        # print(exp_proj_mat)
        # print(eye_mat)
        # print(model_mat)
        # print(model_view_mat)
        # print(proj_mat)
        # print(view_mat)
        program.SetUniform2f("Ext_res", res)
        program.SetUniform3f("Ext_camPos", cam_pos)
        program.SetUniform3f("Ext_focPnt", foc_pnt)
        program.SetUniform3f("Ext_viewUp", view_up)
        program.SetUniformMatrix("Ext_mat", mat)

mapper.AddObserver(vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)


# sphere_actor = actor.sphere(centers=np.zeros(positions.shape),
#                             colors=colors,
#                             radii=radii * 0.5,
#                             theta=8,
#                             phi=8)


# sphere_geometry = numpy_support.vtk_to_numpy(sphere_actor.GetMapper().GetInput().GetPoints().GetData())
centers_geometry = numpy_support.vtk_to_numpy(vtk_centers)
centers_geometryOrig = np.array(numpy_support.vtk_to_numpy(vtk_centers))
centers_length = centers_geometry.shape[0] / positions.shape[0]

verts_geometry = numpy_support.vtk_to_numpy(vtk_verts)
verts_geometryOrig = np.array(numpy_support.vtk_to_numpy(vtk_verts))
verts_length = verts_geometry.shape[0] / positions.shape[0]

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
                        fake_tube=False, linewidth=3,
                        opacity=0.2
                        )

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

    # sphere_geometry = np.array(numpy_support.vtk_to_numpy(
    #     sphere_actor.GetMapper().GetInput().GetPoints().GetData()))
    # geometry_length = sphere_geometry.shape[0] / vertices_count

    # threadID = helios.startAsyncLayout(edgesArray,positions,velocities);
    def iterateHelios(iterationCount):
        for i in range(iterationCount):
            helios.layout(edgesArray,positions,velocities,0.001,0.1,0.02);
        # t1 = threading.Thread(target=heliosLayout, args=(iterationCount,edgesArray,pos,velocities)) 
        # t1.start()
        # t1.join()

    

    framesPerSecond = []
    def _timer(_obj, _event):
        nonlocal counter,framesPerSecond
        counter += 1
        framesPerSecond.append(scene.frame_rate)
        if(counter%100==0):
            print(np.average(framesPerSecond))
            framesPerSecond=[]
        # spheres_positions = numpy_support.vtk_to_numpy(
        #     sphere_actor.GetMapper().GetInput().GetPoints().GetData())
        # spheres_positions[:] = sphere_geometry + \
        #     np.repeat(positions, geometry_length, axis=0)
        # print(sphere_geometry)
        # sphere_geometry[:] = np.repeat(positions, geometry_length, axis=0)
        
        centers_geometry[:] = centers_geometryOrig + \
            np.repeat(positions, centers_length, axis=0)

        verts_geometry[:] = verts_geometryOrig + \
            np.repeat(positions, verts_length, axis=0)
            
        edges_positions = numpy_support.vtk_to_numpy(
            lines_actor.GetMapper().GetInput().GetPoints().GetData())
        edges_positions[::2] = positions[edges_list[:, 0]]
        edges_positions[1::2] = positions[edges_list[:, 1]]

        lines_actor.GetMapper().GetInput().GetPoints().GetData().Modified()
        lines_actor.GetMapper().GetInput().ComputeBounds()
        vtk_verts.Modified()
        vtk_centers.Modified()
        sphere_actor.GetMapper().GetInput().GetPoints().GetData().Modified()
        sphere_actor.GetMapper().GetInput().ComputeBounds()
        # showm.scene.reset_camera()
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

global text_block
text_block = ui.TextBlock2D(font_size=20, bold=True)
text_block.message = ''
text_block.color = (1, 1, 1)

camera = scene.camera()

# scene.add(lines_actor)
# scene.add(sphere_actor)
boxActor = actor.box(centers=np.array([[0,0,0]]),size=(1000,1000,1000),colors=(1,0,0,0.2))
scene.add(boxActor)

###############################################################################
# The final step ! Visualize the result of our creation! Also, we need to move
# the camera a little bit farther from the network. you can increase the
# parameter max_iteractions of the timer callback to let the animation run for
# more time.

showm = window.ShowManager(scene, reset_camera=False, size=(
    1980, 1920), order_transparent=False, multi_samples=2,)


showm.initialize()

# scenePicker.SetRenderer(showm.scene)

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