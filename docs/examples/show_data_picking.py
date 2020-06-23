import numpy as np
from fury import actor, window, ui, utils
from vtk.util import numpy_support
import vtk

fdata = 'xyzr.npy'

#xyzr = np.load(fdata)[:3000, :]
xyzr = np.array([[0, 0, 0, 10], [100, 0, 0, 50], [200, 0, 0, 100]])

colors = np.random.rand(*(xyzr.shape[0], 3))
#colors = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.4], [0, 0, 1., 0.99]])

global text_block
text_block = ui.TextBlock2D(font_size=20, bold=True)
text_block.message = ''
text_block.color = (1, 1, 1)

# panel = ui.Panel2D(position=(150, 90), size=(250, 100),
#                    color=(.6, .6, .6), align="left")
# panel.add_element(text_block, 'relative', (0.2, 0.3))

scene = window.Scene()

sphere_actor = actor.sphere(centers=0.5 * xyzr[:, :3],
                            colors=colors[:],
                            radii=0.1 * xyzr[:, 3])


sphere_geometry = numpy_support.vtk_to_numpy(sphere_actor.GetMapper().GetInput().GetPoints().GetData())
geometry_length = sphere_geometry.shape[0] / len(xyzr)

scene.add(sphere_actor)
scene.reset_camera()

global showm

global picker
# picker = window.vtk.vtkWorldPointPicker()
picker = window.vtk.vtkPropPicker()
scenePicker = vtk.vtkScenePicker()

def left_click_callback(obj, event):

    global text_block, showm, picker,scenePicker
    x, y, z = obj.GetCenter()
    event_pos = showm.iren.GetEventPosition()

    # picker.Pick(event_pos[0], event_pos[1],
    #             0, showm.scene)
    # print(picker.GetPickPosition())
    cell_index = scenePicker.GetCellId(event_pos)
    point_index = scenePicker.GetVertexId(event_pos)
    text = 'Object ID '+str(int(np.floor(point_index/geometry_length)))+"\n"+'Face ID ' + str(cell_index) + '\n' + 'Point ID ' + str(point_index)
    # # text_block.message = text
    print(text)
    # print(cell_index)
    showm.render()

sphere_actor.AddObserver('LeftButtonPressEvent', left_click_callback, 1)

scene.add(actor.axes((10,10,10)))
showm = window.ShowManager(scene, size=(1024, 768), order_transparent=True)
# showm.iren.add_callback()

showm.initialize()

scenePicker.SetRenderer(showm.scene)

# scenePicker.EnableVertexPickingOff()
scenePicker.SetEnableVertexPicking(True)
# scene.add(text_block)

sphere_mapper = sphere_actor.GetMapper()

# sphere_mapper.AddShaderReplacement(
#     vtk.vtkShader.Fragment,
#     '//VTK::Light::Dec',
#     True,
#     '''
#     ERROR
#     ''',
#     False
# )

# #renderer.add(panel)
showm.start()