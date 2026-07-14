from astropy.cosmology import FlatLambdaCDM
from astropy.units import Quantity
from slsim.Sources.SourcePopulation.point_sources import PointSources
from slsim.Sources.SourcePopulation.point_plus_extended_sources import (
    PointPlusExtendedSources,
)
from slsim.Sources.SourceCatalogues.QuasarCatalog.simple_quasar import (
    quasar_catalog_simple,
)
import pytest


@pytest.fixture
def Quasar_class():
    sky_area = Quantity(value=0.1, unit="deg2")
    kwargs_quasars = {
        "num_quasars": 5000,
        "z_min": 0.1,
        "z_max": 5,
        "m_min": 17,
        "m_max": 25,
    }
    quasar_list = quasar_catalog_simple(**kwargs_quasars)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kwargs = {
        "variability_model": "light_curve",
        "kwargs_variability": None,
        "agn_driving_variability_model": None,
        "agn_driving_kwargs_variability": None,
        "lightcurve_time": None,
    }
    return PointSources(
        point_source_list=quasar_list,
        cosmo=cosmo,
        sky_area=sky_area,
        kwargs_cut={},
        point_source_type="quasar",
        joint_point_source_kwargs=kwargs,
    )


def test_source_number(Quasar_class):
    number = Quasar_class.source_number
    assert number > 0


def test_draw_source(Quasar_class):
    quasar = Quasar_class.draw_source()
    assert isinstance(quasar, object)
    assert quasar.redshift > 0

