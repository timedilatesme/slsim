from slsim.FalsePositives.false_positive import FalsePositive
from slsim.Lenses.lens_pop import area_theta_e_infinity
from slsim.LOS.los_pop import LOSPop
from slsim.Lenses.lens_pop import draw_field_galaxies
import random
import numpy as np


class FalsePositivePopBase(object):
    """Base class for false positive population generation.

    This class provides common functionality and can be extended to
    create specific types of false positive populations.
    """

    def __init__(
        self,
        central_galaxy_population,
        cosmo=None,
        los_pop=None,
        field_galaxy_population=None,
    ):
        """
        :param central_galaxy_population: Deflector population as a deflectors class instance.
        :param cosmo: astropy.cosmology instance
        :param los_pop: LOSPop instance which manages line-of-sight (LOS) effects
         and Gaussian mixture models in a simulation or analysis context.
        :param field_galaxy_population: list of field galaxy instances to include in the lensing configuration, if any.
         If provided, these galaxies will be included as additional light in the lens plane, and will not be explicitly included as deflectors in the lensing calculation.
        """
        self.cosmo = cosmo
        self._lens_galaxies = central_galaxy_population
        self._field_galaxy_population = field_galaxy_population
        self.los_config = los_pop or LOSPop()

    def draw_deflector(self):
        """Draw and prepare a deflector (lens) with tolerance-based z_max.

        :return: a deflector instance and z_max for sources.
        """
        deflector = self._lens_galaxies.draw_deflector()
        z_max = deflector.redshift + 0.002  # Adding tolerance to redshift
        return deflector, z_max

    def draw_field_galaxies(self, area, z_max=None):
        """Draw field galaxies within a specified area and redshift limit.

        :param area: Area in which to draw field galaxies (in square
            arcseconds).
        :param z_max: Maximum redshift for the field galaxies. If None,
            no redshift cut is applied.
        :return: List of drawn field galaxy instances.
        """
        return draw_field_galaxies(
            field_galaxy_population=self._field_galaxy_population,
            area=area,
            z_max=z_max,
        )


class FalsePositivePop(FalsePositivePopBase):
    """Class to perform samples of false positive population.

    This class generates configurations consisting of a central
    deflector and one or multiple sources drawn from provided
    populations (e.g., surrounding galaxies, stars, quasars). It
    combines the functionality of single-source and multi-source
    generation with flexible clustering modes.

    Note that these sources are not lensed by the deflector,
    but are positioned in a way that they could be misidentified as
    lensed images of a source due to the deflector, thus creating a "false positive"
    lensing configuration.
    """

    def __init__(
        self,
        central_galaxy_population,
        source_populations,
        source_number_choices=[1, 2, 3],
        weights_for_source_number=None,
        cosmo=None,
        los_pop=None,
        test_area_factor=1,
        clustering_mode="area",
        include_central_galaxy_light=True,
        field_galaxy_population=None,
    ):
        """
        :param central_galaxy_population: Deflector population as a deflectors class instance.
        :param source_populations: A single source population or a list of source populations.
        :param source_number_choices: A list of integers (for a single population) or a list of
               lists containing integers (for multiple populations) representing the possible
               number of sources to draw. Defaults to [1, 2, 3].
        :param weights_for_source_number: Weights corresponding to probabilities for `source_number_choices`.
        Either a single list of weights (for a single population) or a list of lists of weights (for multiple populations).
        :param cosmo: astropy.cosmology instance
        :param los_pop: LOSPop instance.
        :param test_area_factor: A multiplicative factor for the test_area.
        :param clustering_mode: 'area' (default) places sources within the calculated test area,
                                'ring' for ring-like clustering around the deflector (places sources in a ring within a range of 0.5 to 2.5 times the Einstein radius),
        :param include_central_galaxy_light: Whether to include central galaxy light.
        :param field_galaxy_population: Field galaxy population.
        """

        super(FalsePositivePop, self).__init__(
            central_galaxy_population=central_galaxy_population,
            cosmo=cosmo,
            los_pop=los_pop,
            field_galaxy_population=field_galaxy_population,
        )

        # Normalize populations and choices into lists for uniform handling
        if not isinstance(source_populations, list):
            self._source_populations = [source_populations]
            self._number_choices = [source_number_choices]
            self._weights = (
                [weights_for_source_number] if weights_for_source_number else [None]
            )
        else:
            self._source_populations = source_populations
            self._number_choices = source_number_choices
            self._weights = (
                weights_for_source_number
                if weights_for_source_number
                else [None] * len(source_populations)
            )

        # Basic validation
        if len(self._source_populations) != len(self._number_choices):
            raise ValueError(
                "The length of 'source_populations' must match the length of 'source_number_choices'."
            )

        self._clustering_mode = clustering_mode
        self._test_area_factor = test_area_factor
        self._include_central_galaxy_light = include_central_galaxy_light

    def draw_sources(self, z_max, test_area=None, theta_e=None):
        """Draws sources from all populations based on choices and positions
        them.

        :param z_max: maximum redshift for drawn source.
        :param test_area: area to draw source coordinates for 'area'
            clustering mode.
        :param theta_e: Einstein radius of the deflector, needed for
            'ring'/'random' modes.
        :return: A Source instance or a list of Source instances.
        """
        all_sources = []

        # Iterate over each population and its corresponding number choices
        for pop, choices, weights in zip(
            self._source_populations, self._number_choices, self._weights
        ):
            if isinstance(choices, int):
                n_draw = choices
            else:
                n_draw = (
                    random.choices(choices, weights=weights)[0]
                    if weights
                    else random.choice(choices)
                )

            for _ in range(n_draw):
                source = pop.draw_source()
                if source is None:
                    return None
                all_sources.append(source)

        if not all_sources:
            return None

        total_sources = len(all_sources)

        for i, source in enumerate(all_sources):
            if self._clustering_mode == "ring":
                r = random.uniform(0.5 * theta_e, 2.5 * theta_e)
                phi = (2 * np.pi * i / total_sources) + random.uniform(
                    -0.3, 0.3
                )  # Adding some randomness to the angular position
                source.update_center(center_x=r * np.cos(phi), center_y=r * np.sin(phi))
            else:
                # Default "area" positioning architecture
                if test_area is not None:
                    source.update_center(area=test_area)

        return all_sources[0] if len(all_sources) == 1 else all_sources

    def draw_false_positive(self, number=1):
        """Draw given number of false positive systems.

        :param number: number of false positive requested. The default
            value is 1.
        :return: list of FalsePositive() instance (or single instance if
            number=1).
        """
        false_positive_population = []

        for _ in range(number):
            successful = False
            while not successful:
                # Step 1: Draw deflector
                deflector, z_max = self.draw_deflector()

                # Step 2: Calculate areas and sizing for positioning
                theta_e_infinity = deflector.theta_e_infinity(cosmo=self.cosmo)
                test_area = self._test_area_factor * area_theta_e_infinity(
                    theta_e_infinity=theta_e_infinity
                )

                # Step 3: Draw and position sources
                sources = self.draw_sources(
                    z_max=z_max, test_area=test_area, theta_e=theta_e_infinity
                )
                if sources is None:
                    continue  # Retry if sources are invalid

                # Step 4: Add field galaxies
                field_galaxies = self.draw_field_galaxies(
                    area=test_area * 10, z_max=z_max
                )

                # Step 5: Create false positive
                false_positive = FalsePositive(
                    deflector_class=deflector,
                    source_class=sources,
                    cosmo=self.cosmo,
                    include_deflector_light=self._include_central_galaxy_light,
                    los_class=self.los_config.draw_los(
                        source_redshift=z_max, deflector_redshift=deflector.redshift
                    ),
                    field_galaxies=field_galaxies,
                )
                false_positive_population.append(false_positive)
                successful = True

        return (
            false_positive_population[0] if number == 1 else false_positive_population
        )
