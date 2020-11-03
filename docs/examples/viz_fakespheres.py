
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

vertices_count = 500000
view_size = 300

positions = view_size * \
    np.random.random((vertices_count, 3)) - view_size / 2.0

positions = np.ascontiguousarray(positions,dtype=np.float32);

colors = np.array(cmap.cm.viridis(np.arange(0,vertices_count)/(vertices_count-1)))

radii = 1 + np.random.rand(len(positions))


n_points = colors.shape[0]
np.random.seed(42)
centers = positions # np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
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
sphere_actor.GetProperty().BackfaceCullingOff()

# scene.add(canvas_actor)
sphere_actor.GetMapper().SetVBOShiftScaleMethod(False)
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
        program.SetUniform2f("Ext_res", res)
        program.SetUniform3f("Ext_camPos", cam_pos)
        program.SetUniform3f("Ext_focPnt", foc_pnt)
        program.SetUniform3f("Ext_viewUp", view_up)
        program.SetUniformMatrix("Ext_mat", mat)

mapper.AddObserver(vtk.vtkCommand.UpdateShaderEvent, vtk_shader_callback)



scene = window.Scene()

scene.background((1.0, 1.0, 1.0))

camera = scene.camera()

scene.add(sphere_actor)


showm = window.ShowManager(scene, reset_camera=False, size=(
    1980, 1080), order_transparent=False, multi_samples=2,)


showm.initialize()

scene.set_camera(position=(0, 0, -750))

showm.start()

