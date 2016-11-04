import numpy as np

from dipy.viz.interactor import CustomInteractorStyle

from dipy.utils.optpkg import optional_package

# Allow import, but disable doctests if we don't have vtk.
vtk, have_vtk, setup_module = optional_package('vtk')

if have_vtk:
    version = vtk.vtkVersion.GetVTKSourceVersion().split(' ')[-1]
    major_version = vtk.vtkVersion.GetVTKMajorVersion()
else:
    vtkInteractorStyleUser = object


class UI(object):
    """ An umbrella class for all UI elements.
    While adding UI elements to the renderer, we need to go over all the
     sub-elements that come with it and add those to the renderer too.
    There are several features that are common to all the UI elements:
    - ui_param : This is an attribute that can be passed to the UI object
                by the interactor. Thanks to Python's dynamic type-setting
                this parameter can be anything.
    - ui_list : This is used when there are more than one UI elements inside
               a UI element. Inside the renderer, they're all iterated and added.
    - parent_UI: This is useful of there is a parent UI element and its reference
                needs to be passed down to the child.
    """
    def __init__(self):
        self.ui_param = None
        self.ui_list = list()

        self.parent_UI = None
        self._callbacks = []

    def get_actors(self):
        """ Returns the actors that compose this UI component. """
        msg = "Subclasses of UI must implement `get_actors(self)`."
        raise NotImplementedError(msg)

    def add_to_renderer(self, ren):
        """ Allows UI objects to add their own props to the renderer.
        Parameters
        ----------
        ren : renderer
        """
        ren.add(*self.get_actors())

        # Get a hold on the current interactor style.
        iren = ren.GetRenderWindow().GetInteractor().GetInteractorStyle()

        for callback in self._callbacks:
            if not isinstance(iren, CustomInteractorStyle):
                msg = ("The ShowManager requires `CustomInteractorStyle` in"
                       " order to use callbacks.")
                raise TypeError(msg)

            iren.add_callback(*callback, args=[self])

    def add_callback(self, prop, event_type, callback, priority=0):
        """ Adds a callback to a specific event for this UI component.
        Parameters
        ----------
        prop : vtkProp
        event_type : event code
        callback : function
        priority : int
        """
        # Actually since we need an interactor style we will add the callback
        # only when this UI component is added to the renderer.
        self._callbacks.append((prop, event_type, callback, priority))

    def set_center(self, position):
        """ Sets the center of the UI component
        Parameters
        ----------
        position : (float, float)
        """
        msg = "Subclasses of UI must implement `set_center(self, position)`."
        raise NotImplementedError(msg)

    def set_visibility(self, visibility):
        """ Sets visibility of this UI component. """
        for actor in self.get_actors():
            actor.SetVisibility(visibility)


class Button2D(UI):
    """A 2D overlay button and is of type vtkTexturedActor2D.
    Currently supports:
    - Multiple icons.
    - Switching between icons.
    """

    def __init__(self, icon_fnames, size=(30, 30)):
        """
        Parameters
        ----------
        size : 2-tuple of int, optional
            Button size.
        icon_fnames : dict
            {iconname : filename, iconname : filename, ...}
        """
        super(Button2D, self).__init__()
        self.icon_extents = dict()
        self.icons = self.build_icons(icon_fnames)
        self.icon_names = list(self.icons.keys())
        self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]
        self.actor = self.build_actor(self.icons[self.current_icon_name])
        self.size = size

    def build_icons(self, icon_fnames):
        """ Converts file names to vtkImageDataGeometryFilters.
        A pre-processing step to prevent re-read of file names during every state change.
        Parameters
        ----------
        icon_fnames : dict
            {iconname: filename, iconname: filename, ...}
        Returns
        -------
        icons : dict
            A dictionary of corresponding vtkImageDataGeometryFilters
        """
        icons = {}
        for icon_name, icon_fname in icon_fnames.items():
            png = vtk.vtkPNGReader()
            png.SetFileName(icon_fname)
            png.Update()

            texture = vtk.vtkTexture()
            texture.SetInputConnection(png.GetOutputPort())
            icons[icon_name] = texture, png.GetOutput().GetExtent()

        return icons

    @property
    def size(self):
        """Gets the button size."""
        return self._size

    @size.setter
    def size(self, size):
        """Sets the button size.
        Parameters
        ----------
        size : (float, float)
        """
        self._size = np.asarray(size)

        # Update actor.
        self.texture_points.SetPoint(0, 0, 0, 0.0)
        self.texture_points.SetPoint(1, size[0], 0, 0.0)
        self.texture_points.SetPoint(2, size[0], size[1], 0.0)
        self.texture_points.SetPoint(3, 0, size[1], 0.0)
        self.texture_polydata.SetPoints(self.texture_points)

    @property
    def color(self):
        """Gets the button's color."""
        color = self.actor.GetProperty().GetColor()
        return np.asarray(color)

    @color.setter
    def color(self, color):
        """Sets the button's color.
        Parameters
        ----------
        color : (float, float, float)
        """
        self.actor.GetProperty().SetColor(*color)

    def scale(self, size):
        """Scales the button.
        Parameters
        ----------
        size : (float, float)
        """
        self.size *= size

    def build_actor(self, icon):
        """ Return an image as a 2D actor with a specific position.
        Parameters
        ----------
        icon : imageDataGeometryFilter
        Returns
        -------
        button : vtkTexturedActor2D
        """
        # This is highly inspired by
        # https://github.com/Kitware/VTK/blob/c3ec2495b183e3327820e927af7f8f90d34c3474/Interaction/Widgets/vtkBalloonRepresentation.cxx#L47

        self.texture_polydata = vtk.vtkPolyData()
        self.texture_points = vtk.vtkPoints()
        self.texture_points.SetNumberOfPoints(4)
        self.size = icon[1]

        polys = vtk.vtkCellArray()
        polys.InsertNextCell(4)
        polys.InsertCellPoint(0)
        polys.InsertCellPoint(1)
        polys.InsertCellPoint(2)
        polys.InsertCellPoint(3)
        self.texture_polydata.SetPolys(polys)

        tc = vtk.vtkFloatArray()
        tc.SetNumberOfComponents(2)
        tc.SetNumberOfTuples(4)
        tc.InsertComponent(0,0, 0.0);  tc.InsertComponent(0,1, 0.0)
        tc.InsertComponent(1,0, 1.0);  tc.InsertComponent(1,1, 0.0)
        tc.InsertComponent(2,0, 1.0);  tc.InsertComponent(2,1, 1.0)
        tc.InsertComponent(3,0, 0.0);  tc.InsertComponent(3,1, 1.0)
        self.texture_polydata.GetPointData().SetTCoords(tc)

        texture_mapper = vtk.vtkPolyDataMapper2D()
        texture_mapper.SetInputData(self.texture_polydata)

        button = vtk.vtkTexturedActor2D()
        button.SetMapper(texture_mapper)
        button.SetTexture(icon[0])

        button_property = vtk.vtkProperty2D()
        button_property.SetOpacity(1.0)
        button.SetProperty(button_property)

        return button

    def get_actors(self):
        """ Returns the actors that compose this UI component. """
        return [self.actor]

    def add_callback(self, event_type, callback):
        """ Adds events to button actor.
        Parameters
        ----------
        event_type : string
            event code
        callback : function
            callback function
        """
        super(Button2D, self).add_callback(self.actor, event_type, callback)

    def set_icon(self, icon):
        """ Modifies the icon used by the vtkTexturedActor2D.
        Parameters
        ----------
        icon : imageDataGeometryFilter
        """
        self.actor.SetTexture(icon[0])

    def next_icon_name(self):
        """ Returns the next icon name while cycling through icons.
        """
        self.current_icon_id += 1
        if self.current_icon_id == len(self.icons):
            self.current_icon_id = 0
        self.current_icon_name = self.icon_names[self.current_icon_id]

    def next_icon(self):
        """ Increments the state of the Button.
            Also changes the icon.
        """
        self.next_icon_name()
        self.set_icon(self.icons[self.current_icon_name])

    def set_center(self, position):
        """ Sets the icon center to position.
        Parameters
        ----------
        position : (float, float)
        """
        new_position = np.asarray(position) - self.size / 2.
        self.actor.SetPosition(*new_position)
