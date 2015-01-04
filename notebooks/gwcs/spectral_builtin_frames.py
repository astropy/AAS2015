from __future__ import division, print_function
from astropy import units as u
from astropy.units import equivalencies as eq
from astropy.utils import OrderedDict
from astropy import coordinates as coo
from astropy.coordinates import BaseCoordinateFrame, FrameAttribute, TimeFrameAttribute, RepresentationMapping
from astropy.coordinates import BaseRepresentation
from astropy.coordinates import frame_transform_graph

#from astropy import constants

__all__ = ['Wavelength', 'Frequency', 'OpticalVelocity']


class Cartesian1DRepresentation(BaseRepresentation):
    """
    Representation of a one dimensional cartesian coordinate.

    Parameters
    ----------
    x : `~astropy.units.Quantity`
        The coordinate along the axis.

    copy : bool, optional
        If True arrays will be copied rather than referenced.
    """

    attr_classes = OrderedDict([('x', u.Quantity)])

    def __init__(self, x, copy=True):

        if not isinstance(x, self.attr_classes['x']):
            raise TypeError('x should be a {0}'.format(self.attr_classes['x'].__name__))

        x = self.attr_classes['x'](x, copy=copy)

        self._x = x

    @property
    def x(self):
        """
        The x component of the point(s).
        """
        return self._x

    @classmethod
    def from_cartesian(cls, other):
        return other

    def to_cartesian(self):
        return self


class Wavelength(BaseCoordinateFrame):
    default_representation = Cartesian1DRepresentation
    reference_position = FrameAttribute(default='BARYCENTER')
    frame_specific_representation_info = {
        'cartesian1d': [RepresentationMapping('x', 'lambda', 'm')]
        }


class Frequency(BaseCoordinateFrame):
    default_representation = Cartesian1DRepresentation
    reference_position = FrameAttribute(default='BARYCENTER')
    frame_specific_representation_info = {
        'cartesian1d': [RepresentationMapping('x', 'freq', 'Hz')]
        }


class OpticalVelocity(BaseCoordinateFrame):
    default_representation = Cartesian1DRepresentation
    reference_position = FrameAttribute(default='BARYCENTER')
    rest = FrameAttribute()
    frame_specific_representation_info = {
        'cartesian1d': [RepresentationMapping('x', 'v', 'm/s')]
        }


@frame_transform_graph.transform(coo.FunctionTransform, Wavelength, Frequency)
def wave_to_freq(wavecoord, freqframe):
    return Frequency(wavecoord.lam.to(u.Hz, equivalencies=eq.spectral()))


@frame_transform_graph.transform(coo.FunctionTransform, Frequency, Wavelength)
def freq_to_wave(freqcoord, waveframe):
    return Wavelength(freqcoord.f.to(u.m, equivalencies=eq.spectral()))

@frame_transform_graph.transform(coo.FunctionTransform, Wavelength, OpticalVelocity)
def wave_to_velo(wavecoord, veloframe):
    return OpticalVelocity(wavecoord.lam.to(veloframe.representation_component_units.values()[0], equivalencies=eq.doppler_optical(veloframe.rest)))


@frame_transform_graph.transform(coo.FunctionTransform, Frequency, OpticalVelocity)
def freq_to_velo(freqcoord, veloframe):
    return OpticalVelocity(freqcoord.f.to(veloframe.representation_component_units.values()[0], equivalencies=eq.doppler_optical(veloframe.rest)))


