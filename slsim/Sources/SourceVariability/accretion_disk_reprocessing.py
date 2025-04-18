import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy import cosmology
from scipy import signal, interpolate
from slsim.Util.astro_util import (
    calculate_gravitational_radius,
    calculate_accretion_disk_response_function,
    downsample_passband,
    bring_passband_to_source_plane,
    convert_passband_to_nm,
    calculate_accretion_disk_emission,
)
from slsim.Sources.SourceVariability.light_curve_interpolation import (
    LightCurveInterpolation,
)
from speclite.filters import (
    load_filter,
)
from slsim.Util.param_util import (
    magnitude_to_amplitude,
    amplitude_to_magnitude,
)


class AccretionDiskReprocessing(object):
    def __init__(self, reprocessing_model, **kwargs_agn_model):
        """Initialize the accretion disk reprocessing object.

        :param reprocessing_model: keyword for the reprocessing model to be used. Only
            supports "lamppost" now.
        :param kwargs_agn_model: keyword arguments for the variability model. Note that
            these have default values if they are not input. For the lamppost model, the
            kwargs are: ('r_out'), ('r_resolution'), ('inclination_angle'),
            ('black_hole_mass_exponent'), ('black_hole_spin'), ('corona_height'), and
            ('eddington_ratio').
        :type kwargs_agn_model: dict
        """

        self.reprocessing_model = reprocessing_model
        self.kwargs_model = kwargs_agn_model

        if self.reprocessing_model not in ["lamppost"]:
            raise ValueError(
                "Given model is not supported. Currently supported model is lamppost."
            )
        if self.reprocessing_model == "lamppost":
            self._model = lamppost_model

            default_lamppost_kwargs = {
                "r_out": 1000,
                "corona_height": 10,
                "r_resolution": 500,
                "inclination_angle": 0,
                "black_hole_mass_exponent": 8.0,
                "black_hole_spin": 0.0,
                "eddington_ratio": 0.1,
            }

            for jj in default_lamppost_kwargs:
                if jj not in self.kwargs_model:
                    print(
                        "keyword "
                        + jj
                        + " is not defined, using default value of: "
                        + str(default_lamppost_kwargs[jj])
                    )
                    self.kwargs_model[jj] = default_lamppost_kwargs[jj]

            self.time_array = None
            self.magnitude_array = None
        if "redshift" in self.kwargs_model:
            self.redshift = self.kwargs_model["redshift"]
        else:
            self.redshift = 0

    def define_new_response_function(self, rest_frame_wavelength_in_nanometers):
        """Define a response function of the agn accretion disk to the flaring
        corona in the lamppost geometry.

        :param rest_frame_wavelength_in_nanometers: The rest frame
            wavelength (not the observer's frame!) in [nanometers].
        :return: An array representing the response function of the
            accretion disk with time lag spacing of [R_g / c].
        """
        return self._model(rest_frame_wavelength_in_nanometers, **self.kwargs_model)

    def define_passband_response_function(
        self,
        passband,
        redshift=0,
        delta_wavelength=10,
        passband_wavelength_unit=u.angstrom,
    ):
        """Calculates the response function of the agn accretion disk to the
        flaring corona in the lamppost geometry for an input passband.

        :param passband: Str or List representing passband data. Either from speclite or
            a user defined passband represented as a list of lists or arrays. The first
            must be wavelengths, and the second must be the throughput of
            signature: [wavelength, throughput].
        :param redshift: Float value of source redshift. Used to convert wavelengths of
            the passband into emitted wavelengths.
        :param delta_wavelength: Desired wavelength spacing in passband in [nanometers].
            The passband will be resampled to allow for faster calculations.
        :param passband_wavelength_unit: Astropy unit representing the wavelength units
            used to define the original passband. Speclite filters typically use angstroms.
        :return: An array representing the response function of the accretion disk with
            time lag spacing of [R_g / c].
        """
        passband_in_nm = convert_passband_to_nm(
            passband, wavelength_unit_input=passband_wavelength_unit
        )
        passband_in_source_plane = bring_passband_to_source_plane(
            passband_in_nm, redshift
        )
        passband_to_use = downsample_passband(
            passband_in_source_plane,
            delta_wavelength,
            wavelength_unit_input=u.nm,
            wavelength_unit_output=u.nm,
        )
        if len(passband_to_use[0]) > 20:
            print("Warning, over 20 wavelengths to calculate.")
        total_weighting = np.sum(passband_to_use[1])
        total_response_function = (
            self.define_new_response_function(passband_to_use[0][0])
            * passband_to_use[1][0]
            / total_weighting
        )

        for jj in range(len(passband_to_use[0]) - 1):
            if passband_to_use[1][1 + jj] > 0:
                response_function = (
                    self.define_new_response_function(passband_to_use[0][1 + jj])
                    * passband_to_use[1][1 + jj]
                    / total_weighting
                )
                total_response_function += response_function

        return total_response_function

    def define_intrinsic_signal(self, time_array=None, magnitude_array=None):
        """Multi-purpose method to define an intrinsic signal of the
        AccretionDiskReprocessing() class. Passing in the time_array and
        magnitude_array arguments will write the signal to the object, while a
        call with no arguments will return the stored signal.

        :param time_array: The times which the light curve is sampled
            at, in [days].
        :param magnitude_array: The amplitudes of the signal at each
            time in time_array.
        :return: The time_array and magnitude_array associated with the
            AccretionDiskReprocessing() object's intrinsic (driving)
            signal.
        """
        if time_array is None and magnitude_array is None:
            return self.time_array, self.magnitude_array
        if (time_array is None) != (magnitude_array is None):
            raise ValueError(
                "You must provide both the time_array and the magnitude_array."
            )
        if len(time_array) != len(magnitude_array):
            raise ValueError(
                "Input time_array and magnitude_array must be of equal length."
            )
        self.time_array = time_array
        self.magnitude_array = magnitude_array
        return self.time_array, self.magnitude_array

    def reprocess_signal(
        self,
        rest_frame_wavelength_in_nanometers=None,
        response_function_time_lags=None,
        response_function_amplitudes=None,
    ):
        """Multi-purpose method to calculate the response of the accretion disk
        to a reprocessing of the intrinsic signal. Passing the
        rest_frame_wavelength_in_nanometers argument will calculate a response
        function using define_new_response_function(). Passing in a response
        function (e.g. from an external source) to the response_function
        arguments will perform the convolution of the stored intrinsic signal
        with the chosen response function.

        :param rest_frame_wavelength_in_nanometers: Int representing the
            rest frame (not the observer's frame!) wavelength to
            calculate the response function at, in [nanometers].
        :param response_function_time_lags: An optional array
            representing the time_array associated with the response
            function with units [days]. Time lags are defined in the
            rest frame (not the observer's frame!). If None and
            response_function_amplitudes is given, the time lags will be
            assumed to be in units [Rg / c].
        :param response_function_amplitudes: An array representing the
            response function at each time lag. The amplitudes may use
            arbitrary units.
        :return: The magnitude_array of the reprocessed signal. Note
            that this is calculated in the rest frame, not the
            observer's frame!
        """
        if self.time_array is None or self.magnitude_array is None:
            raise ValueError(
                "Please provide the intrinsic signal first, using define_intrinsic_signal()."
            )

        gravitational_radius_in_days = (
            calculate_gravitational_radius(
                self.kwargs_model["black_hole_mass_exponent"]
            )
            / const.c.to(u.m / u.day)
        ).value

        if rest_frame_wavelength_in_nanometers is not None:
            if response_function_amplitudes is not None:
                raise ValueError(
                    "Please provide only a wavelength or only a response function. Not both!"
                )
            response_function = self.define_new_response_function(
                rest_frame_wavelength_in_nanometers
            )
            length_in_days = int(len(response_function) * gravitational_radius_in_days)
            time_lag_axis = np.linspace(0, length_in_days, len(response_function))
        if rest_frame_wavelength_in_nanometers is None:
            if response_function_amplitudes is None:
                raise ValueError("Please provide a wavelength or a response function.")

            response_function = response_function_amplitudes

            if response_function_time_lags is not None:
                if len(response_function_time_lags) != len(
                    response_function_amplitudes
                ):
                    raise ValueError(
                        "The time lag array and response function array must match in length."
                    )

                time_lag_axis = response_function_time_lags
                length_in_days = int(time_lag_axis[-1] - time_lag_axis[0])
            else:
                length_in_days = int(
                    len(response_function) * gravitational_radius_in_days
                )
                time_lag_axis = np.linspace(0, length_in_days, len(response_function))

        if length_in_days == 0:
            length_in_days += 1
            time_lag_axis = np.linspace(0, length_in_days, len(response_function))

        interpolation_of_response_function = interpolate.interp1d(
            time_lag_axis, response_function, bounds_error=False, fill_value=0
        )

        tau_axis = np.linspace(0, int(time_lag_axis[-1]), int(time_lag_axis[-1]) + 1)

        interpolated_response_function = interpolation_of_response_function(tau_axis)

        light_curve = {
            "MJD": np.array(self.time_array),
            "ps_mag_intrinsic": np.array(self.magnitude_array),
        }

        signal_time_axis = np.linspace(
            int(self.time_array[0]),
            int(self.time_array[-1]),
            int(self.time_array[-1]) - int(self.time_array[0]) + 1,
        )

        intrinsic_signal = LightCurveInterpolation(light_curve)
        interpolated_signal = intrinsic_signal.magnitude(signal_time_axis)

        reprocessed_signal = signal.convolve(
            interpolated_signal, (interpolated_response_function), mode="full"
        )

        # bring the reprocessed signal to the observer frame
        interpolation_of_reprocessed_signal = interpolate.interp1d(
            signal_time_axis,
            reprocessed_signal[: len(signal_time_axis)],
            bounds_error=False,
            fill_value=0,
        )
        redshifted_time_axis = signal_time_axis / (1 + self.redshift)
        reprocessed_signal_in_observed_frame = interpolation_of_reprocessed_signal(
            redshifted_time_axis
        )

        normalization = np.nansum(interpolated_response_function)
        if normalization == 0:
            reprocessed_signal_in_observed_frame = intrinsic_signal.magnitude(
                redshifted_time_axis
            )
        else:
            reprocessed_signal_in_observed_frame /= normalization

        return reprocessed_signal_in_observed_frame[: len(self.time_array)]

    def determine_agn_luminosity_from_known_luminosity(
        self,
        known_band,
        known_magnitude,
        redshift,
        mag_zero_point,
        cosmo=cosmology.FlatLambdaCDM(H0=70, Om0=0.3),
        band=None,
        observer_frame_wavelength_in_nm=None,
    ):
        """Takes in the known luminosity of the AGN and defines the expected
        magnitude at other bands or wavelengths based on black body radiation.
        Speclite bands will be calculated at their effective wavelength.

        :param i_band_magnitude: Float representing magnitude of i band
        :param redshift: Float representing redshift of AGN
        :param cosmo: Astropy cosmology object used to calculate
            distances
        :param bands: Float representing a speclite filter
        :param wavelengths: Float representing wavlength in nm.
        """
        if isinstance(known_band, str):
            filter_response = load_filter(known_band)
        else:
            raise ValueError("only speclite filters are implimented")

        effective_wavelength = (filter_response.effective_wavelength).to(u.nm)
        source_plane_wavelength = effective_wavelength / (1 + redshift)
        luminosity_distance = cosmo.luminosity_distance(redshift)
        obs_plane_flux = magnitude_to_amplitude(known_magnitude, mag_zero_point)
        source_plane_flux = luminosity_distance**2 * obs_plane_flux

        theoretical_flux = calculate_accretion_disk_emission(
            self.kwargs_model["r_out"],
            self.kwargs_model["r_resolution"],
            self.kwargs_model["inclination_angle"],
            source_plane_wavelength,
            self.kwargs_model["black_hole_mass_exponent"],
            self.kwargs_model["black_hole_spin"],
            self.kwargs_model["eddington_ratio"],
        )
        # normalize flux
        flux_adjustment_ratio = source_plane_flux / theoretical_flux
        if band is not None:
            band_wavelength = (load_filter(band).effective_wavelength).to(u.nm) / (
                1 + redshift
            )
            cur_theoretical_flux = (
                calculate_accretion_disk_emission(
                    self.kwargs_model["r_out"],
                    self.kwargs_model["r_resolution"],
                    self.kwargs_model["inclination_angle"],
                    band_wavelength,
                    self.kwargs_model["black_hole_mass_exponent"],
                    self.kwargs_model["black_hole_spin"],
                    self.kwargs_model["eddington_ratio"],
                )
                * flux_adjustment_ratio
            )
            cur_obs_flux = cur_theoretical_flux / luminosity_distance**2
            output_magnitude = amplitude_to_magnitude(cur_obs_flux, mag_zero_point)
        elif observer_frame_wavelength_in_nm is not None:
            source_plane_wavelength = observer_frame_wavelength_in_nm / (1 + redshift)
            cur_theoretical_flux = (
                calculate_accretion_disk_emission(
                    self.kwargs_model["r_out"],
                    self.kwargs_model["r_resolution"],
                    self.kwargs_model["inclination_angle"],
                    source_plane_wavelength,
                    self.kwargs_model["black_hole_mass_exponent"],
                    self.kwargs_model["black_hole_spin"],
                    self.kwargs_model["eddington_ratio"],
                )
                * flux_adjustment_ratio
            )
            cur_obs_flux = cur_theoretical_flux / luminosity_distance**2
            output_magnitude = amplitude_to_magnitude(cur_obs_flux, mag_zero_point)
        else:
            raise ValueError("Please define a band or wavelength")
        return output_magnitude.value


def lamppost_model(
    rest_frame_wavelength_in_nanometers,
    r_out=1000,
    r_resolution=500,
    inclination_angle=0,
    black_hole_mass_exponent=8.0,
    black_hole_spin=0.0,
    corona_height=10,
    eddington_ratio=0.1,
):
    """Uses astro_util.calculate_accretion_disk_response_function() to define
    the response of the accretion disk due to a driving signal in the lamppost
    geometry.

    :param rest_frame_wavelength_in_nanometers: The wavelength to calculate the response
        function at, in [nanometers]. This is in the frame of the accretion disk, not the
        frame of the observer.
    :param r_out: Maximum radius of the accretion disk in gravitational radii [R_g =
        GM/c^2], typically 10^3 to 10^5.
    :param r_resolution: Number of gridpoints to use between 0 and r_out. Final map will
        be calculated at (2*r_resolution) x (2*r_resolution).
    :param inclination_angle: The tilt of the accretion disk with respect to the
        observer [degrees]
    :param black_hole_mass_exponent: The log of the black hole mass normalized by the
        mass of the sun; black_hole_mass_exponent = log_10(black_hole_mass / mass_sun).
        Typical AGN have an exponent ranging from 6 to 10.
    :param black_hole_spin: The dimensionless spin parameter of the black hole, where
        the spinless case represents the Schwarzschild black hole. Maximum values of +/-
        1.
    :param corona_height: The height of the corona lamppost in gravitational radii [R_g].
        Typical choices range from 0 to 100.
    :param eddington_ratio: The desired Eddington ratio defined as a fraction of
        bolometric luminosity / Eddington luminosity.
    :return: The response function of the accretion disk with respect to variability in
        the corona.
    """

    return calculate_accretion_disk_response_function(
        r_out=r_out,
        r_resolution=r_resolution,
        inclination_angle=inclination_angle,
        rest_frame_wavelength_in_nanometers=rest_frame_wavelength_in_nanometers,
        black_hole_mass_exponent=black_hole_mass_exponent,
        black_hole_spin=black_hole_spin,
        corona_height=corona_height,
        eddington_ratio=eddington_ratio,
    )
