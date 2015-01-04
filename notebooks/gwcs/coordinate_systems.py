# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Defines coordinate frames and ties them to data axes.
"""
import numpy as np
from astropy.utils.compat.numpy import broadcast_arrays
from astropy.utils import OrderedDict
from astropy import time
from astropy import units as u
from astropy import utils as astutil
from astropy import coordinates as coo
#from astropy.coordinates.representation import CartesianRepresentation
#from astropy.coordinates.baseframe import BaseCoordinateFrame#, frame_transform_graph
from astropy.coordinates import (BaseRepresentation, BaseCoordinateFrame, FrameAttribute,
                                 RepresentationMapping)

from astropy import units as u
from . import spectral_builtin_frames
from .spectral_builtin_frames import *

__all__ = ['DetectorFrame', 'CelestialFrame', 'SpectralFrame', 'TimeFrame',
           'CompositeFrame', 'FocalPlaneFrame']


celestial_frames = coo.builtin_frames.__all__[:]
spectral_frames = spectral_builtin_frames.__all__[:]

class Cartesian2DRepresentation(BaseRepresentation):
    """
    Representation of a two dimensional cartesian coordinate system

    Parameters
    ----------
    x : `~astropy.units.Quantity`
        The coordinate along the X axis.
    y : `~astropy.units.Quantity`
        The coordinate along the Y axis.
    copy : bool, optional
        If True arrays will be copied rather than referenced.
    """

    attr_classes = OrderedDict([('x', u.Quantity),
                                ('y', u.Quantity)])

    def __init__(self, x, y, copy=True):

        if not isinstance(x, self.attr_classes['x']):
            raise TypeError('x should be a {0}'.format(self.attr_classes['x'].__name__))
        if not isinstance(y, self.attr_classes['y']):
            raise TypeError('y should be a {0}'.format(self.attr_classes['y'].__name__))

        x = self.attr_classes['x'](x, copy=copy)
        y = self.attr_classes['y'](y, copy=copy)

        if not (x.unit.physical_type == y.unit.physical_type):
            raise u.UnitsError("x and y should have matching physical types")

        try:
            x, y = broadcast_arrays(x, y, subok=True)
        except ValueError:
            raise ValueError("Input parameters x and y cannot be broadcast")

        self._x = x
        self._y = y

    @property
    def x(self):
        """
        The x component of the point(s).
        """
        return self._x

    @property
    def y(self):
        """
        The y component of the point(s).
        """
        return self._y

    @classmethod
    def from_cartesian(cls, other):
        return other

    def to_cartesian(self):
        return self


class CoordinateFrame(object):
    """
    Base class for CoordinateFrames

    Parameters
    ----------
    naxes : int
        number of axes
    ref_system : str
        reference system (see subclasses)
    ref_pos : str or ``ReferencePosition``
        reference position or standard of rest
    units : list of units
        unit for each axis
    axes_names : list
        names of the axes in this frame, in thee order of the data axes.
    name : str
        name (alias) of this frame
    """
    def __init__(self, naxes, axes_mapping=(0, 1), reference_frame=None, reference_position=None,
                 unit=None, axes_names=None, name=None):
        """ Initialize a frame"""
        self._axes_mapping = axes_mapping
        # map data axis into frame axes - 0-based
        self._naxes = naxes
        if unit is not None:
            if astutil.isiterable(unit):
                if len(unit) != naxes:
                    raise ValueError("Number of units does not match number of axes")
                else:
                    self._unit = [u.Unit(au) for au in unit]
            else:
                self._unit = u.Unit(unit)
        else:
            self._unit = reference_frame.representation_component_units.values()

        if axes_names is not None and astutil.isiterable(axes_names):
            if len(axes_names) != naxes:
                raise ValueError("Number of axes names does not match number of axes")
        else:
            axes_names = reference_frame.representation_component_names.values()
        self._axes_names = axes_names
        self._reference_frame = reference_frame
        self._reference_position = reference_position
        if name is None:
            self._name = reference_frame.name
        else:
            self._name = name
        if reference_position is not None:
            self._reference_position = reference_position
        else:
            try:
                self._reference_position = reference_frame.reference_position
            except AttributeError:
                self._reference_position = None

        super(CoordinateFrame, self).__init__()

    def __repr__(self):
        if self._name is not None:
            return self._name
        else:
            return ""

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return ""

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = name

    @property
    def naxes(self):
        return self._naxes

    @property
    def unit(self):
        return self._unit

    @property
    def axes_names(self):
        return self._axes_names

    @property
    def reference_frame(self):
        return self._reference_frame

    @property
    def reference_position(self):
        try:
            return self._reference_position
        except AttributeError:
            return None

    @property
    def axes_mapping(self):
        return self._axes_mapping

    def transform_to(self, other):
        """
        Transform from the current reference system to other
        """
        raise NotImplementedError("Subclasses should implement this")


    #def _validate_reference_position(self):
        #assert self._reference_position in self.standard_reference_position, "Unrecognized reference position"

class TimeFrame(CoordinateFrame):
    """
    Time Frame

    Parameters
    ----------
    scale : str
        time scale, one of ``standard_time_scale``
    format : str
        time representation
    obslat : float or ``astropy.coordinates.Latitude``
        observer's latitude
    obslon : float or ``astropy.coordinates.Longitude``
        observer's longitude
     units : str or ``astropy.units.Unit``
        unit for each axis
    axes_names : list of str
        names of the axes in this frame
    """

    standard_time_scale = time.Time.SCALES

    time_format = time.Time.FORMATS

    def __init__(self, scale, format, obslat=0., obslon=0., units="s", axes_names="Time", name="Time"):
        assert scale in self.standard_time_scale, "Unrecognized time scale"
        assert format in self.time_format, "Unrecognized time representation"
        super(TimeFrame, self).__init__(1, units=[units], axes_names=[axes_names], name=name)
        self._scale = scale
        self._format = format
        self._obslat = obslat
        self._obslon = obslon

    def transform_to(self, val, tosys, val2=None, precision=None, in_subfmt=None,
                     out_subfmt=None, copy=False):
        """Transforms to a different scale/representation"""

        t = time.Time(val, val2, format=self._format, scale=self._scale,
                      precision=precision, in_subfmt=in_subfmt, out_subfmt=out_subfmt,
                      lat=self._obslat, lon=self._obslon)
        return getattr(t, tosys)


class CelestialFrame(CoordinateFrame):
    """
    Space Frame Representation

    Parameters
    ----------
    reference_system : str
        one of standard_ref_frame
    reference_position : str, ReferencePosition instance
    obstime : time.Time instance
        observation time
    equinox : time.Time
        equinox
    ##TODO? List of supported projections per frame
    projection : str
        projection type
    axes_names : iterable or str
        axes labels
    units : str or units.Unit instance or iterable of those
        units on axes
    distance : Quantity
        distance from origin
        see ``~astropy.coordinates.distance`
    obslat : float or ~astropy.coordinates.Latitute`
        observer's latitude
    obslon : float or ~astropy.coordinates.Longitude
        observer's longitude

    """
    standard_ref_frame = ["FK4", "FK5", "ICRS", '...']

    standard_reference_positions = ["GEOCENTER", "BARYCENTER", "HELIOCENTER",
                                    "TOPOCENTER", "MOON", "EMBARYCENTER", "RELOCATABLE",
                                    "UNKNOWNRefPos"]

    def __init__(self, reference_frame, axes_mapping=(0, 1), reference_position=None,
                 unit=[u.degree, u.degree], name=""):
        #, obstime, equinox=None,
        #                 projection="", axes_names=["", ""],  distance=None,
        #                 obslat=None, obslon=None, name=None):
        # when the frame is realized, this is the unit to use.
        #unit = unit
        reference_position = 'Barycenter' #??
        if reference_frame.name.upper() in celestial_frames:
            axes_names = reference_frame.representation_component_names.keys()[:2]
        super(CelestialFrame, self).__init__(naxes=2, reference_frame=reference_frame,
                                             unit=unit, axes_mapping= axes_mapping,
                                             reference_position=reference_position,
                                             axes_names=axes_names, name=name)

    @property
    def obstime(self):
        return self.reference_frame.obstime

    @property
    def equinox(self):
        return self.reference_frame.equinox

    def world_coordinates(self, lon, lat):
        return coo.SkyCoord(lon, lat, unit=self.unit, frame=self._reference_frame)

    def __repr__(self):
        ref = repr(self.reference_frame)
        return "{0}, {1}, axes_mapping={2}, axes_names={3}".format(ref, self.unit,
                                                                   self.axes_mapping,
                                                                   self.axes_names)

    def transform_to(self, lat, lon, other):
        """
        Transform from the current reference system to other.
        """
        if other.name in celestial_frames:
            return self.world_coordinates(lon, lat).transform_to(other)
    '''
        func = self._wrap_func(lat, lon, other, distance)
        return func(lat, lon)

    def _wrap_func(self, lat, lon, other, distance):
        def create_coordinates(lat, lon, distance):
            coord_klass = getattr(coo, self._reference_system)
            current_coordinates = coord_klass(lat, lon, unit=self.units, distance=distance, obstime=self._obstime, equinox=self._equinox)
            return current_coordinates.transform_to(other)
        return create_coordinates
    '''

class SpectralFrame(CoordinateFrame):
    """
    Represents Spectral Frame

    Parameters
    ----------
    reference_frame : str
        one of spectral_ref_frame
    reference_position : str
        one of standard_reference position
    unit : str or units.Unit instance
        spectral unit
    axes_names : str
        spectral axis name
        If None, reference_system is used
    rest_freq : float?? or None or Quantity
        rest frequency in units

    obstime : time.Time instance
    obslat : float or instance of `~astropy.coordinates.Latitude`
        observer's latitude
    obslon : float or instanceof `~astropy.coordinates.Longitude`
        observer's longitude
    """
    #spectral_ref_frame = ["WAVE", "FREQ", "ENER", "WAVEN",  "AWAV", "VRAD",
                          #"VOPT", "ZOPT", "VELO", "BETA"]

    #standard_reference_position = ["GEOCENTER", "BARYCENTER", "HELIOCENTER",
                                   #"TOPOCENTER", "LSR", "LSRK", "LSRD",
                                   #"GALACTIC_CENTER", "MOON", "LOCAL_GROUP_CENTER"]

    def __init__(self, reference_frame, unit, axes_mapping=(0,), axes_names=None,
                 name="", rest_frequency=None):

        if axes_names is None:
            axes_names = reference_frame.representation_component_names.values()
        if name == "":
            name = reference_frame.name
        super(SpectralFrame, self).__init__(naxes=1, axes_mapping=axes_mapping,
                                            reference_frame=reference_frame,
                                            axes_names=axes_names, unit=unit, name=name)

    def __repr__(self):
        return "{0}, {1}, axes_mapping={2}, axes_names={3}".format(self.reference_frame, self.unit,
                                                                   self.axes_mapping, self.axes_names)

    def transform_to(self, x, other):
        """
        Transform from the current reference system to other.
        """
        if other.name in spectral_frames:
            return self(x).transform_to(other)


class CompositeFrame(CoordinateFrame):
    """
    Represents one or more frames.

    Parameters
    ----------
    name : str
        a user defined name
    frames : list
        list of frames
        one of TimeFrame, CelestialFrame, SpectralFrame, CoordinateFrame, CoordinateSystem
    """
    def __init__(self, frames, name=""):
        """
        frames : list
            coordinate frames
        name : str
            name of the coordinate system WCSNAME?
        reference : dict
            keywords specifying the reference position of this observation in this
            coordinate system
        """
        self._frames = frames[:]
        naxes = sum([frame._naxes for frame in self._frames])
        unit = []
        axes_mapping = []
        axes_names = []
        for frame in frames:
            unit.extend(frame.unit)
            axes_mapping.extend(frame.axes_mapping)
            axes_names.extend(frame.axes_names)
        #for frame in self._frames[1:]:
            #unit.extend(frame.unit)
        #self._units = units
        #self._name = name
        super(CompositeFrame, self).__init__(naxes, axes_mapping=axes_mapping,
                                             unit=unit, axes_names=axes_names,
                                             name=name)
    @property
    def unit(self):
        return [frame.unit for frame in self.frames]

    @property
    def frames(self):
        return self._frames

    def __repr__(self):
        return repr(self.frames)

#class CartesianFrame(CoordinateFrame):
    #def __init__(self, naxes, units, name=None, axes_names=None, reference_position=None):
        ###TODO: reference position
        #super(CartesianFrame, self).__init__(naxes, ref_pos=reference_position, units=units, axes_names=axes_names, name=name)

    #def transform_to(self, other, *args):
        #pass

class DetectorFrame(CoordinateFrame):
    def __init__(self, reference_frame, axes_mapping=(0, 1), name='Detector', reference_position="Local"):
        axes_names = ['x', 'y']
        super(DetectorFrame, self).__init__(2, name=name, axes_names=axes_names,
                                            reference_position=reference_position)


class Detector(BaseCoordinateFrame):
    default_representation = Cartesian2DRepresentation
    reference_position = FrameAttribute(default='Local') ##?
    reference_point = FrameAttribute()
    frame_specific_representation_info = {
        'cartesian2d': [RepresentationMapping('x', 'x', 'pixel'),
                        RepresentationMapping('y', 'y', 'pixel')]
        }

class FocalPlaneFrame(CoordinateFrame):
    def __init__(self, reference_pixel=[0., 0.], units=[u.pixel, u.pixel], name='FocalPlane',
                 reference_position="Local", axes_names=None):
        super(FocalPlaneFrame, self).__init__(2, reference_position=reference_position, units=units, name=name, axes_names=axes_names)
        self._reference_pixel = reference_pixel

class ImageFrame(CoordinateFrame):
    def __init__(self, reference_pixel= [0., 0.], units=[u.pixel, u.pixel], name='Image',
                 reference_position="Local", axes_names=None):
        super(DetectorFrame, self).__init__(2, reference_position=reference_position, units=units, name=name, axes_names=axes_names)
        self._reference_pixel = reference_pixel

class ReferencePosition(object):

    standard_reference_position = ["GEOCENTER", "BARYCENTER", "HELIOCENTER",
                                   "TOPOCENTER", "LSR", "LSRK", "LSRD",
                                   "GALACTIC_CENTER", "MOON", "EMBARYCENTER",
                                   "RELOCATABLE", "UNKNOWNRefPos"]

    def __init__(self, ref_pos, lat=None, lon=None, alt=None):
        if not ref_pos in standard_reference_position:
            raise ValueError("Unrecognized reference position")

        if ref_pos == 'TOPOCENTRIC' and (lat is None or lon is None or alt is None):
            raise ValueError("Need lat, lon, alt to define Topocentric reference position")

        self._ref_pos = ref_pos

    @property
    def ref_pos(self):
        return self._ref_pos


