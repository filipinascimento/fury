import os
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk
import multiprocessing
import numpy as np
# import imagezmq


class FuryStreamClient:
    def __init__(
            self, showm,
            window_size=(200, 200),
            write_in_stdout=False,
            broker_url=None):
        '''

        Parameters
        ----------
            showm: fury showm manager
        '''
        self.showm = showm
        self.window2image_filter = vtk.vtkWindowToImageFilter()
        self.window2image_filter.SetInput(self.showm.window)
        self.write_in_stdout = write_in_stdout
        self.image_buffer = multiprocessing.RawArray(
            'B', np.random.randint(
                0, 255, size=window_size[0]*window_size[1]*3).astype('uint8'))

        self.info_buffer = multiprocessing.RawArray(
            'I', np.array([window_size[0], window_size[1], 3]))

        self._id_timer = None
        self._id_observer = None
        self.sender = None
        self._in_request = False
        if broker_url is not None:
            pass
            # self.sender = imagezmq.ImageSender(connect_to=broker_url)

    def init(self, ms=16):
        if self.write_in_stdout:
            os.system('cls' if os.name == 'nt' else 'clear')

        window2image_filter = self.window2image_filter

        def callback(caller, timerevent):
            if not self._in_request:
                self._in_request = True
                self.window2image_filter.Update()
                self.window2image_filter.Modified()
                vtk_image = window2image_filter.GetOutput()
                vtk_array = vtk_image.GetPointData().GetScalars()
                np_arr = vtk_to_numpy(vtk_array).astype('uint8')

                h, w, _ = vtk_image.GetDimensions()
                num_components = vtk_array.GetNumberOfComponents()

                if self.image_buffer is not None:
                    if self.info_buffer is not None:
                        self.info_buffer[0] = h
                        self.info_buffer[1] = w
                        self.info_buffer[2] = num_components
                    np_arr = np_arr.flatten()
                    self.image_buffer[:] = np_arr
                self._in_request = False

        if ms > 0:
            id_timer = self.showm.add_timer_callback(
                True, ms, callback)
            self._id_timer = id_timer
        else:
            id_observer = self.showm.iren.AddObserver(
                'RenderEvent', callback)
            self._id_observer = id_observer

    def stop(self):
        if self._id_timer is not None:
            self.showm.destroy_timer(self._id_timer)

        if self._id_observer is not None:
            self.showm.iren.RemoveObserver(self._id_observer)


class FuryStreamInteraction:
    def __init__(self, showm, circular_queue, max_size=5, dimension=4):
        self.showm = showm
        self.recorder = vtk.vtkInteractorEventRecorder()
        self.recorder.SetInteractor(self.showm.iren)

        self.circular_queue = circular_queue
        self._id_timer = None

    def start(self, ms=16):
        def callback(caller, timerevent):
            data = self.circular_queue.dequeue()
            if data is not None:
                if data[0] == 1:
                    zoomFactor = 1.0 - data[1] / 100.0
                    # camera = showm.window
                    # .GetRenderers().GetFirstRenderer().GetActiveCamera()
                    camera = self.showm.scene.GetActiveCamera()
                    fp = camera.GetFocalPoint()
                    pos = camera.GetPosition()
                    delta = [fp[i] - pos[i] for i in range(3)]
                    camera.Zoom(zoomFactor)

                    pos2 = camera.GetPosition()
                    camera.SetFocalPoint(
                        [pos2[i] + delta[i] for i in range(3)])
                    self.showm.iren.LeftButtonPressEvent()
                    self.showm.iren.LeftButtonReleaseEvent()

                elif data[0] == 2:
                    # print(self.showm.iren.GetLastEventPosition())
                    self.showm.iren.LeftButtonPressEvent()
                    self.showm.iren.SetControlKey(int(data[4]))
                    self.showm.iren.SetShiftKey(int(data[5]))

                    self.showm.iren.SetEventPosition(
                        int(self.showm.size[0]*data[2]),
                        int(self.showm.size[1]*data[3])
                    )

                    self.showm.iren.MouseMoveEvent()

                elif data[0] == 3:
                    self.showm.iren.LeftButtonReleaseEvent()

            self.showm.render()

        self._id_timer = self.showm.add_timer_callback(True, ms, callback)

    def stop(self):
        if self._id_timer is not None:
            self.showm.destroy_timer(self._id_timer)
