import pytest
import numpy as np
from unittest.mock import patch
from astropy.cosmology import FlatLambdaCDM
import slsim.Sources as sources
import slsim.Deflectors as deflectors
import slsim.Pipelines as pipelines
from slsim.FalsePositives.false_positive_pop import (
    FalsePositiveGalaxiesPop,
    FalsePositiveMultiSourcePop
)
from astropy.units import Quantity

sky_area = Quantity(value=0.01, unit="deg2")
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
    skypy_config=None,
    sky_area=sky_area,
    filters=None,
)
kwargs_deflector_cut = {"band": "g", "band_max": 28, "z_min": 0.01, "z_max": 2.5}
kwargs_source_cut = {"band": "g", "band_max": 28, "z_min": 0.1, "z_max": 5.0}
red_galaxy_list = galaxy_simulation_pipeline.red_galaxies
blue_galaxy_list = galaxy_simulation_pipeline.blue_galaxies

lens_galaxies = deflectors.EllipticalLensGalaxies(
    galaxy_list=red_galaxy_list,
    kwargs_cut=kwargs_deflector_cut,
    kwargs_mass2light=0.1,
    cosmo=cosmo,
    sky_area=sky_area,
)
kwargs = {"extended_source_type": "single_sersic"}
source_galaxies = sources.Galaxies(
    galaxy_list=blue_galaxy_list,
    kwargs_cut=kwargs_source_cut,
    cosmo=cosmo,
    sky_area=sky_area,
    catalog_type="skypy",
    **kwargs
)


def test_draw_false_positive_single():
    fp_pop1 = FalsePositiveGalaxiesPop(
        central_galaxy_population=lens_galaxies,
        surrounding_galaxy_population=source_galaxies,
        cosmo=cosmo,
        source_number_choice=[1],
    )
    draw_fp1 = fp_pop1.draw_false_positive()
    draw_deflector = fp_pop1.draw_deflector()
    draw_source = fp_pop1.draw_sources(z_max=draw_deflector[1])
    
    assert isinstance(draw_fp1, object)
    assert isinstance(draw_deflector[0], object)
    assert draw_deflector[1] == draw_deflector[0].redshift + 0.002
    assert isinstance(draw_source, object)


def test_draw_false_positive_multiple():
    fp_pop2 = FalsePositiveGalaxiesPop(
        central_galaxy_population=lens_galaxies,
        surrounding_galaxy_population=source_galaxies,
        cosmo=cosmo,
        source_number_choice=[2],
    )
    draw_fp2 = fp_pop2.draw_false_positive(number=2)
    assert isinstance(draw_fp2, list)


@patch('numpy.random.poisson', return_value=2)
def test_draw_false_positive_with_field_galaxies(mock_poisson):
    # Tests the base class method draw_field_galaxies through the population generator
    fp_pop = FalsePositiveGalaxiesPop(
        central_galaxy_population=lens_galaxies,
        surrounding_galaxy_population=source_galaxies,
        cosmo=cosmo,
        source_number_choice=[1],
        field_galaxy_population=source_galaxies # using source_galaxies as mock
    )
    draw_fp = fp_pop.draw_false_positive()
    
    # We forced poisson to draw 2 field galaxies
    assert draw_fp._field_galaxies is not None
    assert len(draw_fp._field_galaxies) == 2


def test_false_positive_multi_source_validation():
    with pytest.raises(ValueError):
        # Mismatched lengths should throw an error
        FalsePositiveMultiSourcePop(
            central_galaxy_population=lens_galaxies,
            source_populations=[source_galaxies],
            source_number_choices=[[1], [2]], 
            cosmo=cosmo
        )


def test_false_positive_multi_source_random_clustering():
    fp_multi_pop = FalsePositiveMultiSourcePop(
        central_galaxy_population=lens_galaxies,
        source_populations=[source_galaxies, source_galaxies],
        source_number_choices=[[0, 1], [1]], # Tests the n_draw == 0 skip logic as well
        cosmo=cosmo,
        clustering_mode="random"
    )
    draw_fp = fp_multi_pop.draw_false_positive()
    assert isinstance(draw_fp, object)
    # The first list can yield 0 or 1, the second yields 1. Total = 1 or 2.
    assert draw_fp.source_number in [1, 2]


def test_false_positive_multi_source_ring_clustering():
    fp_multi_pop = FalsePositiveMultiSourcePop(
        central_galaxy_population=lens_galaxies,
        source_populations=[source_galaxies],
        source_number_choices=[[3]],
        cosmo=cosmo,
        clustering_mode="ring"
    )
    
    draw_fp = fp_multi_pop.draw_false_positive()
    assert isinstance(draw_fp, object)
    assert draw_fp.source_number == 3

    # Check if they have been assigned coordinates correctly
    for i in range(draw_fp.source_number):
        center_source = draw_fp.source(i)._source._center_source
        x, y = center_source[0], center_source[1]
        r = np.sqrt(x**2 + y**2)
        theta_e = draw_fp.einstein_radius_infinity
        # For ring clustering, sources should be roughly around the Einstein radius (0.5 to 2.5 times theta_e)
        assert 0.5 * theta_e < r < 2.5 * theta_e


if __name__ == "__main__":
    pytest.main()