import functools
import copy
import numpy as np
from astropy.io import fits
from astropy import modeling
from astropy.modeling.core import Model
from astropy.modeling import models as astmodels
from astropy.modeling.parameters import Parameter
#from . import util
from . import util


class WCSLinear(object):
    """
    A convenience object to create a WCS linear transformation.

    Parameters
    ----------
    translation : astropy.Modeling.Shift
    rotation: astropy.modeling.AffineTransformation
    scaling : astropy.modeling.Scale

    Returns
    -------
    model : astropy.modeling.CompoundModel

    """
    def __new__(cls, translation=None, rotation=None, scaling=None):
        # add a check for dimensions
        transforms = [translation, rotation, scaling]
        for transform in transforms[::-1]:
            if transform is None:
                transforms.remove(transform)
            if not isinstance(transform, modeling.Model):
                raise util.UnsupportedTransformError("transforms must be an instance of modeling.Model")
        return functools.reduce(lambda x, y: x | y, transforms)

    @classmethod
    def from_fits(cls, header):
        if not isinstance(header, fits.Header):
            raise TypeError("expected a FIST Header")
        fitswcs = util.read_wcs_from_header(header)
        wcsaxes = fitswcs['WCSAXES']

        if fitswcs['has_cd']:
            pc = fitswcs['CD']
        else:
            pc = fitswcs['PC']
        # get the part of the PC matrix corresponding to the imaging axes
        sky_axes = None
        if pc.shape != (2, 2):
            sky_axes, _ = util.get_axes(fitswcs)
            i, j = sky_axes
            sky_pc = np.zeros((2,2))
            sky_pc[0, 0] = pc[i, i]
            sky_pc[0, 1] = pc[i, j]
            sky_pc[1, 0] = pc[j, i]
            sky_pc[1, 1] = pc[j, j]
            pc = sky_pc.copy()

        if sky_axes is not None:
            crpix = []
            cdelt = []
            for i in sky_axes:
                crpix.append(fitswcs['CRPIX'][i])
                cdelt.append(fitswcs['CDELT'][i])
        else:
            cdelt = fitswcs['CDELT']
            crpix = fitswcs['CRPIX']
        translation = astmodels.Shift(-crpix[0], name='offset_x') & astmodels.Shift(-crpix[1], name='offset_y')
        rotation = astmodels.AffineTransformation2D(matrix=pc, name='orient')
        scale = astmodels.Scale(cdelt[0], name='scale_x') & astmodels.Scale(cdelt[1], name='scale_y')
        return cls(translation, rotation, scale)


class ImagingWCS(object):
    """
    A convenience object to concatenate a distortion transformation with
    WCSLinear transformation, projection and sky rotation.

    """
    def __new__(cls,  wcs_linear, projection=None, sky_rotation=None, distortion=None):
        # add a check for dimensions
        transforms = [distortion, wcs_linear, projection, sky_rotation]
        for transform in transforms[::-1]:
            if transform is None:
                transforms.remove(transform)
            if not isinstance(transform, modeling.Model):
                raise util.UnsupportedTransformError("transforms must be an instance of modeling.Model")
        if transforms != []:
            return functools.reduce(lambda x, y: x | y, transforms)
        else:
            return None

    @classmethod
    def from_file(cls, header, dist_json):
        """
        header : fits.Header
        dist_json : json file

        """
        if not isinstance(header, fits.Header):
            raise TypeError("Header must be a fits.Header object")
        fitswcs = util.read_wcs_from_header(header)

        wcs_linear = WCSLinearTransform.from_header(header)
        projection = create_projection_transform(fitswcs)

        # Create a RotateNative2Celestial transform using the Euler angles
        phip, lonp = fitswcs['CRVAL']
        # Expand this - currently valid for zenithal projections only
        thetap = 180 # write "def compute_lonpole(projcode, l)"
        n2c = astmodels.RotateNative2Celestial(phip, lonp, thetap)
        if dist_json is not None:
            distortion = transform_from_json(dist_info['regions'][0])
        else:
            distortion = None

        return cls(wcs_linear, projection, n2c, distortion)


class CompositeSpectralWCS(object):
    """
    A convenience object to join sky and spectral transforms.

    Parameters
    ----------
    sky_transform : Model
        ImagingWCS or some other transform
    spectral_transform : Model
        SpectralWCS or other callable

    """

    def __new__(cls, sky_transform, spectral_transform, spectral_axes=(0,)):
        transforms = [sky_transform, spectral_transform]
        return functools.reduce(lambda x, y: x & y, transforms)

    #def __init__(self, sky_transform=None, spectral_transform=None):#, spectral_inmap):
        #self.sky_transform = sky_transform
        #self.spectral_transform = spectral_transform
        ##self.spectral_inmap = spectral_inmap

    @classmethod
    def from_json(cls, wcs_info=None, spec_info=None, dist_info=None):
        # This should be handled through the coordinate frames but until
        # they work, do it here
        sky_inmap, spec_inmap = util.get_axes(wcs_info)
        sky_wcs_info = util.get_sky_wcs_info(wcs_info)
        sky_transform = ImagingWCS.from_file(wcs_info=sky_wcs_info, dist_info=dist_info)
        spectral_transform = transform_from_json(spec_info['regions'][0])
        return cls(sky_transform, spectral_transform)

    def undistort(self, x, y):
        input_values = np.asarray(x), np.asarray(y)
        return self.sky_transform.undistort(x, y)

    def __call__(self, *args, **kwargs):
        """
        args is x, y
        """
        x, y = args

        sky_res = (None, None)
        spec_res = None
        if self.sky_transform is not None:
            sky_res = self.sky_transform(x, y)
        if self.spectral_transform is not None:
            spec_res = self.spectral_transform(x, y)
        if self.sky_transform.n_outputs == 1:
            return sky_res, spec_res
        elif self.sky_transform.n_outputs == 2:
            #if sky_res[0].ndim == 1:
            return sky_res[0], sky_res[1], spec_res

        if np.isscalar(sky_res):
            result = list([sky_res])
        else:
            result = list(sky_res)
        result.append(spec_res)
        return tuple(result)


def create_projection_transform(wcsinfo):

    projcode = util.get_projcode(wcsinfo['CTYPE'])
    projklassname = 'Pix2Sky_' + projcode
    projklass = getattr(projections, projklassname)
    projparams={}
    return projklass(**projparams)

def transform_from_json(reg_models_json, schema=None):
    """
    reg_models_json : json object
        One region from a json file, e.g.
        reg_models_json['regions'][0]

    Assumes this was validated.
    """
    forward_transform = reg_models_json['forward_transform']
    if len(forward_transform) == 1:
        ft = forward_transform[0]
        #return SerialCompositeModel(chain_transforms(ft))
        return chain_transforms(ft)
    '''
    else:
        transform_list = []
        for output_axis in forward_transform:
            transform_list.append(SerialCompositeModel(chain_transforms(output_axis)))
        return SeparableTransform(transform_list)
    '''

def chain_transforms(transform_json):
    """
    transform_json comes from a json reference file and is a list of
    models for one output coordinate in a specific direction, with inputs
    and ooutputs information.
    e.g. create forward transformation with input "x" and "y" and
    output "alpha" coordinate for a MIRI IFU slice.

    """
    #dict_transforms = OrderedDict()
    transforms_list = []
    transforms = copy.deepcopy(transform_json)
    for model in transforms['models']:
        model_name = model.pop('model_name')
        inputs = model.pop('input', None)
        outputs = model.pop('output', None)
        if inputs is not None and len(inputs) == 1:
            inputs = (inputs[0], )
        if outputs is not None and len(outputs) == 1:
            outputs = (outputs[0],)
        mclass = getattr(astmodels, model_name)
        try:
            model_inst = mclass(**model)
        except:
            raise
        #dict_transforms[model_inst] = [inputs, outputs]
        transforms_list.append(model_inst)
    #return dict_transforms#, inmap
    if transforms_list != []:
        return functools.reduce(lambda x, y: x | y, transforms_list)
    else:
        return None

