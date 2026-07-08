from slsim.Deflectors.deflector_group import DeflectorGroup
from slsim.Deflectors.deflector import Deflector
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM

import numpy.testing as npt
import copy


class TestDeflectorGroup(object):

    def setup_method(self):
        kwargs_light = {"extended_source_type": "single_sersic",
                        "n_sersic": 1,
                        "e1": 0.1,
                        "e2": 0.2,
                        "angular_size": 0.5,
                        "mag_r": 25.,
                        "mag_g": 20.,
                        "stellar_mass": 1e11}
        kwargs_mass = {"mass_type": "EPL",
                       "vel_disp": 250,
                       "gamma_pl": 2,
                       "e1": 0.1,
                       "e2": -0.1,
                       }
        center_x = 0
        center_y = 0
        z = 0.1
        self.deflector = Deflector(z=z, center_x=center_x, center_y=center_y, kwargs_light=kwargs_light,
                                   kwargs_mass=kwargs_mass)
        self.deflector_group = DeflectorGroup(z=z, kwargs_mass_list=[kwargs_mass], kwargs_light_list=[kwargs_light],
                                         center_x_deflector_list=[0], center_y_deflector_list=[0],
                                              center_x=center_x, center_y=center_y)
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.lens_cosmo = LensCosmo(z_lens=z, z_source=1.5, cosmo=self.cosmo)

    def test_deflector_type(self):
        assert self.deflector_group.deflector_type == "group"

    def test_redshift(self):
        assert self.deflector.redshift == self.deflector_group.redshift

    def test_velocity_dispersion(self):
        npt.assert_almost_equal(self.deflector_group.velocity_dispersion(deflector_index=0),
                                self.deflector.velocity_dispersion(), decimal=1)

    def test_deflector_center(self):
        npt.assert_almost_equal(self.deflector_group.deflector_center, self.deflector.deflector_center, decimal=5)

    def test_update_center(self):
        deflector_group_copy = copy.deepcopy(self.deflector_group)
        deflector_group_copy.update_center(area=1)
        _, kwargs_mass_model = deflector_group_copy.mass_model_lenstronomy(lens_cosmo=self.lens_cosmo)
        assert kwargs_mass_model[0]["center_x"] != 0

    def test_mass_model_lenstronomy(self):
        model_list, kwargs_list = self.deflector_group.mass_model_lenstronomy(lens_cosmo=self.lens_cosmo)
        model_list_, kwargs_list_ = self.deflector.mass_model_lenstronomy(lens_cosmo=self.lens_cosmo)
        assert model_list[0] == model_list_[0]
        npt.assert_almost_equal(kwargs_list_[0]["theta_E"], kwargs_list[0]["theta_E"], decimal=5)

    def test_light_model_lenstronomy(self):
        model_list, kwargs_list = self.deflector_group.light_model_lenstronomy(band="g")
        model_list_, kwargs_list_ = self.deflector.light_model_lenstronomy(band="g")
        assert model_list[0] == model_list_[0]
        npt.assert_almost_equal(kwargs_list_[0]["R_sersic"], kwargs_list[0]["R_sersic"], decimal=5)

    def test_surface_brightness(self):
        ra, dec = 1., 0.2
        band = "g"
        mag_arcsec = self.deflector_group.surface_brightness(ra, dec, band=band)
        mag_arcsec_ = self.deflector.surface_brightness(ra, dec, band=band)
        npt.assert_almost_equal(mag_arcsec, mag_arcsec_, decimal=5)

    def test_theta_e_infinity(self):

        theta_e = self.deflector_group.theta_e_infinity(self.cosmo, use_jax=True)
        theta_e_ = self.deflector.theta_e_infinity(self.cosmo, use_jax=True)
        # this can't be more precise as Ds/Dds even for very high source redshift does not approach =1
        npt.assert_almost_equal(theta_e/theta_e_, 1, decimal=1)