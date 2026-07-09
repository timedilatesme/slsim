from slsim.Deflectors import deflector_util
from slsim.Deflectors.deflector import Deflector
from astropy.cosmology import FlatLambdaCDM


def test_deflector_from_table():

    mass_type = "EPL"
    extended_source_type = "single_sersic"
    table = {"z": 0.2,
        "angular_size": 1, "n_sersic": 1, "e1_light": 0.1, "e2_light": 0.1, "stellar_mass": 1e11,
             "vel_disp": 250, "e1_mass": -0.1, "e2_mass": 0.1}
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    deflector = deflector_util.deflector_from_table(table, mass_type, extended_source_type, cosmo=cosmo)
    assert isinstance(deflector, Deflector)
