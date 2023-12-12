
import jax
import jax.numpy as jnp
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
from herculens.LensImage.lens_image import LensImage

npix = 101 # number of pixel on a side
pix_scl = 0.03  # pixel size in arcsec
half_size = npix * pix_scl / 2
ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2  # position of the (0, 0) with respect to bottom left pixel
transform_pix2angle = pix_scl * jnp.eye(2)  # transformation matrix pixel <-> angle
kwargs_pixel = {'nx': npix, 'ny': npix,
                'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                'transform_pix2angle': transform_pix2angle}

# create the PixelGrid class
pixel_grid = PixelGrid(**kwargs_pixel)

# simple PSF
psf = PSF(psf_type='GAUSSIAN', fwhm=0.12)

# source galaxy
kwargs_pixelated_source = {
    'pixel_scale_factor': 0.03,
    #TODO: maybe this is wrong? 
    'grid_center': (jnp.asarray([half_size]),jnp.asarray([half_size])),
    'grid_shape': (3.03,3.03), # arcsec
}
source_model_pixelated = LightModel(['PIXELATED'], kwargs_pixelated=kwargs_pixelated_source)

lens_mass_model_input = MassModel(['SIE'])

kwargs_numerics_simu = {'supersampling_factor': 1}

lens_image_simu = LensImage(pixel_grid, psf,
                        lens_mass_model_class=lens_mass_model_input,
                        source_model_class=source_model_pixelated,
                        kwargs_numerics=kwargs_numerics_simu)


def apply_lensing(im1: jnp.ndarray, im2: jnp.ndarray, lens_input: jnp.ndarray) -> (jnp.ndarray,jnp.ndarray):
#def apply_lensing(im1,im2,lens_input):

    kwargs_lens_input = [{'theta_E': lens_input[0], 'e1': lens_input[1], 
        'e2': lens_input[2], 'center_x': lens_input[3], 'center_y': lens_input[4]}]
    lensed_im1 = lens_image_simu.source_surface_brightness([{'pixels':im1}], kwargs_lens_input, 
        unconvolved=False, supersampled=False,
        k=[True], k_lens=[True]) 
    
    # wrap up all parameters for the lens_image.model() method
    #model_params = dict(kwargs_lens=[{'theta_E': lens_input[0], 'e1': lens_input[1], 
    #    'e2': lens_input[2], 'center_x': lens_input[3], 'center_y': lens_input[4]}], 
    #    kwargs_source=[{'pixels':im1}])
    # generates the model image
    #lensed_im1 = lens_image_simu.model(**model_params)

    #model_params = dict(kwargs_lens=[{'theta_E': lens_input[0], 'e1': lens_input[1], 
    #    'e2': lens_input[2], 'center_x': lens_input[3], 'center_y': lens_input[4]}], 
    #    kwargs_source=[{'pixels':im2}])
    #lensed_im2 = lens_image_simu.model(**model_params)
    lensed_im2 = lens_image_simu.source_surface_brightness([{'pixels':im2}], kwargs_lens_input, 
        unconvolved=False, supersampled=False,
        k=[True], k_lens=[True]) 
    return lensed_im1,lensed_im2

def batched_apply_lensing(original_ims: jnp.ndarray, decoded_ims: jnp.ndarray) -> (jnp.ndarray,jnp.ndarray):
#def batched_apply_lensing(original_ims,decoded_ims):
    
    # TODO: have to figure out how to loop here

    lens_input = jnp.array([1.,0.,0.,0.0,0.])
    lens_input = lens_input[jnp.newaxis,...]
    batched_lens_input = jnp.repeat(lens_input,original_ims.shape[0],axis=0)
    #batched_lens_input = jnp.tile(lens_input,original_ims.shape[0])
    #lens_input = [{'theta_E': 1.3, 'e1': 0.01, 'e2': -0.05, 'center_x': 0., 'center_y': 0.}]

    # there are too many if/else statements, how to cope? (we can't vmap...)

    lensed_originals = []
    lensed_decoded = []
    for b in range(0,original_ims.shape[0]):
        lensedim1,lensedim2 = apply_lensing(original_ims[b],decoded_ims[b],batched_lens_input[b])
        lensed_originals.append(lensedim1)
        lensed_decoded.append(lensedim2)

    #apply_lensing_vmap = jax.vmap(apply_lensing)
    #output = apply_lensing_vmap(original_ims,decoded_ims,batched_lens_input)

    return jnp.asarray(lensed_originals),jnp.asarray(lensed_decoded)
    #return lensed_original_ims, lensed_decoded_ims

