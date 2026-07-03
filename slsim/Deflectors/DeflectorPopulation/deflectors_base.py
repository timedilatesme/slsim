import numpy as np

from slsim.Sources.SourcePopulation.galaxies import Galaxies, galaxy_projected_eccentricity
from slsim.Sources.SourcePopulation.galaxies import _convert_catalog_to_source
from slsim.Deflectors.deflector import Deflector


class DeflectorsBase(Galaxies):
    """Abstract Base Class to create a class that accesses a set of deflectors.

    All object that inherit from Lensed System must contain the methods
    it contains.
    """

    def __init__(
        self,
        deflector_table,
        kwargs_cut,
        cosmo,
        sky_area,
        gamma_pl=None,
        mass_type="EPL",
        light_type="single_sersic",
        kwargs_mass2light=None,
        catalog_type=None,
    ):
        """

        :param deflector_table: table with lens parameters
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param cosmo: astropy.cosmology instance
        :type sky_area: `~astropy.units.Quantity`
        :param sky_area: Sky area (solid angle) over which galaxies are sampled.
        :param mass_type: type of Deflector() mass model class
        :type mass_type: string
        :param light_type: type of Source() model class for the light distribution
        :type light_type: string
        :param gamma_pl: power law slope in EPL profile.
        :type gamma_pl: A float or a dictionary with given mean and standard deviation
         of a density slope for gaussian distribution or minimum and maximum values of
         gamma for uniform distribution. eg: gamma_pl=2.1, gamma_pl={"mean": a, "std_dev": b},
         gamma_pl={"gamma_min": c, "gamma_max": d}
        :param kwargs_mass2light: mass-to-light relation
        :param catalog_type: type of the catalog. If user is using deflector catalog
         other than generated from skypy pipeline, we require them to provide angular
         size of the galaxy in arcsec and specify catalog_type as None. Otherwise, by
         default, this class considers deflector catalog is generated using skypy
         pipeline.
        """
        super().__init__(galaxy_list=deflector_table,
                        cosmo=cosmo,
                        sky_area=sky_area,
                        kwargs_cut=kwargs_cut,
                        catalog_type=catalog_type,
                        size_model=None,
                        extended_source_type=light_type,
                        downsample_to_dc2=False,
                        extended_source_kwargs=None,
                         )

        self.mass_type = mass_type
        self._kwargs_mass2light = kwargs_mass2light

        # TODO: should be an individual routine
        # set the power-law slope
        galaxy_number = len(deflector_table)
        self._gamma_pl = gamma_pl


    def deflector_number(self):
        """

        :return: number of deflectors after applied cuts
        """
        return self.source_number_selected


    def draw_deflector(self, z_max=None, z_min=None, galaxy_index=None):
        """

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param z_min: minimum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :param galaxy_index: index of galaxy to pic (if provided)
        :return: dictionary of complete parameterization of a deflector
        """

        halo_gal = self.draw_galaxy(z_max=z_max, z_min=z_min, galaxy_index=galaxy_index)

        kwargs_source = _convert_catalog_to_source(galaxy=halo_gal, extended_source_type=self._extended_source_type,
                                                   catalog_type=self._catalog_type, size_model=self._size_model,
                                                   cosmo=self._cosmo, include_all_keywords=False)
        kwargs_mass = self._light2mass(kwargs_source, halo_dict=halo_gal, **self._kwargs_mass2light)
        z = kwargs_source.pop("z")
        deflector_class = Deflector(z=z, kwargs_mass=kwargs_mass, kwargs_light=kwargs_source)
        return deflector_class

    def _light2mass(self, kwargs_source, light2mass_e_scaling=1, light2mass_e_scatter=0.1, halo_dict=None):
        """

        :param kwargs_source: dictioinary for Source() class
        :param light2mass_e_scaling: scaling factor of mass eccentricity /
        light eccentricity
        :param light2mass_e_scatter: scatter in light and mass
            eccentricities from the scaling relation
        :param light2mass_angle_scatter: scatter in orientation angle
            between light and mass eccentricity
        :param halo_dict: dictionary of halo
        :type halo_dict: entry of object to act as deflector
        :return: dictionary for Mass() class
        """
        if halo_dict is not None:
            halo_columns = halo_dict.colnames
        else:
            halo_columns = []
        kwargs_mass = {"mass_type": self.mass_type}
        if self.mass_type in ["EPL", "NFW", "NFW_HERNQUIST"]:

            if "e1_mass" in halo_columns and "e2_mass" in halo_columns:
                e1_mass, e2_mass = halo_dict["e1_mass"], halo_dict["e2_mass"]
            elif "e_h" in halo_columns and "p_h" in halo_columns:
                # SL_hammock conventions
                e1_mass, e2_mass = galaxy_projected_eccentricity(
                    float(halo_dict["e_h"]), rotation_angle=np.deg2rad(halo_dict["p_h"])
                )
            else:
                # scale light to mass ellipticity
                e1_light, e2_light = kwargs_source["e1"], kwargs_source["e1"]
                e1_mass, e2_mass = light2mass_e_scaling * e1_light, light2mass_e_scaling * e2_light
                # add scatter in mass
                e_mass_scatter = np.random.normal(loc=0, scale=light2mass_e_scatter)
                phi_scatter = np.random.uniform(0, np.pi)
                e1_mass += e_mass_scatter * np.cos(2 * phi_scatter)
                e2_mass += e_mass_scatter * np.sin(2 * phi_scatter)
            kwargs_mass["e1"] = e1_mass
            kwargs_mass["e2"] = e2_mass

        if self.mass_type in ["EPL"]:
            gamma_pl = _gamma_pl(self._gamma_pl)
            kwargs_mass["gamma_pl"] = gamma_pl
            kwargs_mass["vel_disp"] = kwargs_source["vel_disp"]
        if self.mass_type in ["NFW", "NFW_HERNQUIST"]:
            if halo_dict is None:
                raise ValueError("halo_dict needs to be provided for mass_type %s" % self.mass_type)
            kwargs_mass["halo_mass"] = halo_dict["halo_mass"]
            kwargs_mass["concentration"] = halo_dict["concentration"]


        return kwargs_mass

def _gamma_pl(gamma_pl):
    """
    madd density power-law slope (2 = isothermal)

    :param gamma_pl:
    :return:
    """
    if gamma_pl is not None:
        if isinstance(gamma_pl, float):
            return gamma_pl
        elif isinstance(gamma_pl, dict):
            parameters = gamma_pl.keys()
            if "mean" in parameters and "std_dev" in parameters:
                return np.random.normal(
                    loc=gamma_pl["mean"],
                    scale=gamma_pl["std_dev"]
                )
            elif "gamma_min" in parameters and "gamma_max" in parameters:
                return np.random.uniform(
                    low=gamma_pl["gamma_min"],
                    high=gamma_pl["gamma_max"],
                )
            else:
                raise ValueError(
                    "The given quantities in gamma_pl are not recognized."
                    " Please provide the mean and standard deviation for a"
                    " gaussian distribution, or specify the gamma_min and gamma_max "
                    " for a uniform distribution."
                )
        else:
            raise ValueError(
                "The given format of the gamma_pl is not supported."
                " Please provide a float or dictionary. See the documentation"
                " in DeflectorsBase class"
            )
    return 2