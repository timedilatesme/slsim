from slsim.Lenses.LensPopulation.lenses_from_catalog import LensPopCatalog
from slsim.Lenses.lens import Lens
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM


class TestLensPopCatalog(object):

    def setup_method(self):
        lens_table = Table(
            names=(
                "z_deflector",
                "z_source",
                "center_x_deflector",
                "center_y_deflector",
                "center_x_source",
                "center_y_source",
                "mag_i_deflector",
                "mag_i_source",
                "angular_size_deflector",
                "angular_size_source",
                "n_sersic_deflector",
                "n_sersic_source",
                "e1_deflector",
                "e2_deflector",
                "e1_source",
                "e2_source",
                "vel_disp_deflector",
                "gamma_1_los",
                "gamma_2_los",
                "kappa_los",
            ),
            rows=[
                (
                    0.5,
                    1.5,
                    0.01,
                    -0.01,
                    0.1,
                    -0.1,
                    17,
                    19,
                    1.5,
                    0.5,
                    1,
                    4,
                    -0.1,
                    -0.1,
                    0.1,
                    0.2,
                    250,
                    0.01,
                    -0.02,
                    0.03,
                )
            ],
        )
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        self.lens_pop_catalog = LensPopCatalog(lens_catalog=lens_table, cosmo=cosmo)

    def test_lens_from_table(self):

        lens = self.lens_pop_catalog.lens_from_table(index=0)
        assert isinstance(lens, Lens)