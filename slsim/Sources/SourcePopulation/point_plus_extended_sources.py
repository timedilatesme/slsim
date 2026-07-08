from slsim.Sources.source import Source
from slsim.Sources.SourcePopulation.galaxies import Galaxies
from slsim.Lenses.selection import object_cut


class PointPlusExtendedSources(Galaxies):
    """Class to describe population of point + extended sources."""

    def __init__(
        self,
        point_plus_extended_sources_list,
        cosmo,
        sky_area,
        kwargs_cut=None,
        catalog_type=None,
        size_model=None,
        point_source_type=None,
        extended_source_type=None,
        point_source_kwargs={},
    ):
        """

        :param point_plus_extended_sources_list: list of dictionary with point and
         extended source parameters or astropy table of sources.
        :param cosmo: cosmology
        :type cosmo: ~astropy.cosmology class
        :param sky_area: Sky area over which galaxies are sampled. Must be in units of
            solid angle.
        :type sky_area: `~astropy.units.Quantity`
        :param kwargs_cut: cuts in parameters: band, band_mag, z_min, z_max
        :type kwargs_cut: dict
        :param catalog_type: type of the catalog. If someone wants to use scotch
         catalog, they need to specify it.
        :type catalog_type: str. eg: "scotch" or None
        :param size_model: If "Bernardi", computes galaxy size using g-band
         magnitude otherwise rescales skypy source size to Shibuya et al. (2015):
         https://iopscience.iop.org/article/10.1088/0067-0049/219/2/15/pdf
        :param point_source_type: Keyword to specify type of the point source.
         Supported point source types are "supernova", "quasar", "general_lightcurve".
        :param extended_source_type: keyword for number of sersic profile to use in source
         light model. accepted kewords: "single_sersic", "double_sersic".
        :param point_source_kwargs: dictionary of keyword arguments for PointSource.
         For supernova kwargs dict, please see documentation of SupernovaEvent class.
         For quasar kwargs dict, please see documentation of Quasar class.
         Eg of supernova kwargs: point_source_kwargs={
         "variability_model": "light_curve", "kwargs_variability": ["supernovae_lightcurve",
            "i", "r"], "sn_type": "Ia", "sn_absolute_mag_band": "bessellb",
            "sn_absolute_zpsys": "ab", "lightcurve_time": np.linspace(-50, 100, 150),
            "sn_modeldir": None}.
        :param extended_source_kwargs: dictionary of keyword arguments for ExtendedSource.
         Please see documentation of ExtendedSource() class as well as specific extended source classes.
        """
        object_list = object_cut(
            point_plus_extended_sources_list,
            object_type="point",
            **kwargs_cut
        )
        Galaxies.__init__(
            self,
            galaxy_list=object_list,
            cosmo=cosmo,
            sky_area=sky_area,
            kwargs_cut={},
            catalog_type=catalog_type,
            size_model=size_model,
            extended_source_type=extended_source_type,
            # extended_source_kwargs=extended_source_kwargs,
        )

        self._point_source_kwargs = point_source_kwargs
        self._point_source_type = point_source_type

    def draw_source(self, z_max=None, z_min=None, galaxy_index=None):
        """Choose source at random.

        :param z_max: maximum redshift limit for the galaxy to be drawn.
            If no galaxy is found for this limit, None will be returned.
        :return: instance of Source class
        """
        kwargs_source = self.draw_source_dict(z_max=z_max, z_min=z_min, galaxy_index=galaxy_index,
                                              include_all_keywords=True)
        if kwargs_source is None:
            return None
        source_class = Source(
            cosmo=self._cosmo,
            point_source_type=self._point_source_type,
            **self._point_source_kwargs,
            **kwargs_source
        )
        return source_class
