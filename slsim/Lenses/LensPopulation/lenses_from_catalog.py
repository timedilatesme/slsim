from typing import Optional
from astropy.units import Quantity
from astropy.cosmology import Cosmology
import random
from slsim.Deflectors.deflector_util import deflector_from_table
from slsim.Lenses.lens import Lens
from slsim.Sources.source import Source
from slsim.LOS.los_individual import LOSIndividual


class LensPopCatalog(object):
    """Class to deal with pre-established set of lenses.

    The catalog shoudl contain parameters for the deflector ending with
    <_deflector>, parameters for the source ending with <_source> and
    for the line of sight ending with <_los>.
    """

    def __init__(
        self,
        lens_catalog,
        cosmo: Optional[Cosmology] = None,
        sky_area: Optional[float or Quantity] = None,
        catalog_type=None,
        use_jax=True,
        deflector_mass_type="EPL",
        deflector_light_type="single_sersic",
        extended_source_type="single_sersic",
        point_source_type=None,
    ):
        """

        :param lens_catalog: catalog with all deflector and source parameters with <_source>, <_deflector> and <_los>
         added
        :param cosmo: astropy.cosmology instance
        :param sky_area: Sky area (solid angle) over which Lens population is sampled.
        :type sky_area: `~astropy.units.Quantity`
        :param catalog_type: catalog type for special conversions of catalog entries ot SLSim conventions
        :type catalog_type: str
        :param use_jax: if True, will use JAX version of lenstronomy to do lensing calculations for models that are
         supported in JAXtronomy
        :type use_jax: bool
        :param deflector_mass_type: deflector mass type consistent with Deflector() class and catalog input
        :param deflector_light_type: deflector light model type consistent with Source() class and catalog input
        """
        self._lens_catalog = lens_catalog
        self._num_lenses = len(lens_catalog)
        self._catalog_type = catalog_type
        self._use_jax = use_jax
        self._cosmo = cosmo
        self._sky_area = sky_area
        self._deflector_mass_type = deflector_mass_type
        self._deflector_light_type = deflector_light_type
        self._point_source_type = point_source_type
        self._extended_source_type = extended_source_type

    def lens_from_table(self, index=None):
        """

        :param index:
        :return: Lens() class
        """
        if index is None:
            index = random.randint(0, self._num_lenses)
        lens_object = self._lens_catalog[index]

        deflector_dict, source_dict, los_dict = _catalog_deflector_source_split(
            lens_object
        )
        deflector = deflector_from_table(
            deflector_dict,
            mass_type=self._deflector_mass_type,
            extended_source_type=self._deflector_light_type,
            cosmo=self._cosmo,
        )
        source = Source(
            extended_source_type=self._extended_source_type,
            point_source_type=self._point_source_type,
            **source_dict
        )
        los_class = LOSIndividual(**los_dict)
        lens_class = Lens(
            source_class=source,
            deflector_class=deflector,
            cosmo=self._cosmo,
            los_class=los_class,
        )
        return lens_class


def _catalog_deflector_source_split(lens_object):
    """Split catalog with <_source> and <_deflector>

    :param lens_object: single column of catalog
    :return: deflector_dict, source_dict, los_dict
    """
    if isinstance(lens_object, dict):
        colnames = list(lens_object.keys())
    else:
        colnames = lens_object.colnames

    deflector_dict = {}
    source_dict = {}
    los_dict = {}
    for item in colnames:
        if item.endswith("_deflector"):
            clean_item = item.removesuffix("_deflector")
            deflector_dict[clean_item] = lens_object[item]
        elif item.endswith("_source"):
            clean_item = item.removesuffix("_source")
            source_dict[clean_item] = lens_object[item]
        elif item.endswith("_los"):
            clean_item = item.removesuffix("_los")
            los_dict[clean_item] = lens_object[item]
    return deflector_dict, source_dict, los_dict
