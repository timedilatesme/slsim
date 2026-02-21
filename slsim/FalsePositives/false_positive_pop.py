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


class FalsePositiveGalaxiesPop(FalsePositivePopBase):
    """Class to perform samples of false positive population.

    Here, false positives refer to a configuration that includes an
    elliptical galaxy at the center with blue galaxies surrounding the
    central elliptical galaxy. This class generates specified number of
    false positives.
    """

    def __init__(
        self,
        central_galaxy_population,
        surrounding_galaxy_population,
        cosmo=None,
        los_pop=None,
        source_number_choice=[1, 2, 3],
        weights_for_source_number=None,
        test_area_factor=1,
        include_central_galaxy_light=True,
        field_galaxy_population=None,
    ):
        """
        Args:
        :param central_galaxy_population: Deflector population as a deflectors class
         instance.
        :param surrounding_galaxy_population: Surrounding galaxy population as a sources class instance.
        :param cosmo: astropy.cosmology instance
        :param los_pop: LOSPop instance which manages line-of-sight (LOS) effects
         and Gaussian mixture models in a simulation or analysis context.
        :param source_number_choice: A list of integers to choose source number from. If
         None, defaults to [1, 2, 3].
        :param weights_for_source_number: A list of weights corresponding to the probabilities of
         selecting each value in source_number_choice. If None, all choices are equally
         likely. Defaults to None.
        :param test_area_factor: A multiplicative factor of a test_area. A test area is
         computed using a velocity dispersion of a central galaxy and that area is
         multiplied by this factor. A default value is 1.
        :param include_central_galaxy_light: Whether to include central galaxy light in the false positive configuration. Default is True.
        """

        super(FalsePositiveGalaxiesPop, self).__init__(
            central_galaxy_population=central_galaxy_population,
            cosmo=cosmo,
            los_pop=los_pop,
            field_galaxy_population=field_galaxy_population,
        )

        self._sources = surrounding_galaxy_population
        self._choice = source_number_choice
        self._weights = weights_for_source_number
        self._test_area_factor = test_area_factor
        self._include_central_galaxy_light = include_central_galaxy_light

    def draw_sources(self, z_max, area=None):
        """Draw source(s) within the redshift limit of z_max.

        :param z_max: maximum redshift for drawn source.
        :param area: area to draw source coordinates, if None, does not
            draw it
        :return: A Source instance or a list of Source instance.
        """
        source_number = random.choices(self._choice, weights=self._weights)[0]
        source_list = []

        for _ in range(source_number):
            source = self._sources.draw_source(z_max=z_max)
            # If no source is available, return None
            if source is None:
                return None
            if area is not None:
                source.update_center(area=area)
            source_list.append(source)
        if source_number == 1:
            sources = source_list[0]
        else:
            sources = source_list
        return sources

    def draw_false_positive(self, number=1):
        """Draw given number of false positives within the cuts of the lens and
        source.

        :param number: number of false positive requested. The default
            value is 1.
        :return: list of FalsePositive() instance.
        """
        false_positive_population = []

        for _ in range(number):
            successful = False
            while not successful:
                # Step 1: Draw deflector
                deflector, z_max = self.draw_deflector()
                # Step 2: Draw sources
                theta_e_infinity = deflector.theta_e_infinity(cosmo=self.cosmo)
                test_area = self._test_area_factor * area_theta_e_infinity(
                    theta_e_infinity=theta_e_infinity
                )
                source = self.draw_sources(z_max, area=test_area)
                if source is None:
                    continue  # Retry if sources are invalid

                # Step 3: Draw field galaxies
                field_galaxies = self.draw_field_galaxies(
                    area=test_area * 10, z_max=z_max
                )

                # Step 4: Create false positive
                false_positive = FalsePositive(
                    deflector_class=deflector,
                    source_class=source,
                    cosmo=self.cosmo,
                    los_class=self.los_config.draw_los(
                        source_redshift=z_max, deflector_redshift=deflector.redshift
                    ),  # Draw LOS for each false positive
                    include_deflector_light=self._include_central_galaxy_light,
                    field_galaxies=field_galaxies,
                )
                false_positive_population.append(false_positive)
                successful = True
        return (
            false_positive_population[0] if number == 1 else false_positive_population
        )


class FalsePositiveMultiSourcePop(FalsePositivePopBase):
    """Class to perform samples of false positive populations with multiple
    source types.

    This class generates configurations consisting of a central
    deflector and multiple point sources drawn from provided populations
    (e.g. stars, quasars).
    """

    def __init__(
        self,
        central_galaxy_population,
        source_populations,
        source_number_choices,
        cosmo=None,
        clustering_mode="random",
        test_area_factor=1,
        los_pop=None,
        include_central_galaxy_light=True,
        field_galaxy_population=None,
    ):
        """
        :param central_galaxy_population: Deflector population as a deflectors class instance.
        :param source_populations: A list of source populations.
        :param source_number_choices: A list of lists. Each inner list contains integers
               representing the possible number of sources to draw from the corresponding population.
               Example: [[1, 2], [4]] means:
               - Draw 1 OR 2 sources from population 1.
               - Draw exactly 4 sources from population 2.
        :param cosmo: astropy.cosmology instance
        :param clustering_mode: 'random' for chance alignments in a box, 'ring' for beads-on-a-string alignment within radius of 0.5 to 2.5 times Einstein Radius.
        :param test_area_factor: A multiplicative factor of a test_area.
        """

        super(FalsePositiveMultiSourcePop, self).__init__(
            central_galaxy_population=central_galaxy_population,
            cosmo=cosmo,
            los_pop=los_pop,
            field_galaxy_population=field_galaxy_population,
        )

        self._source_populations = source_populations
        self._number_choices = source_number_choices
        self._clustering_mode = clustering_mode
        self._test_area_factor = test_area_factor
        self._include_central_galaxy_light = include_central_galaxy_light

        # Basic validation
        if len(self._source_populations) != len(self._number_choices):
            raise ValueError(
                "The length of 'source_populations' must match the length of 'source_number_choices'."
            )

    def draw_sources(self, z_max, theta_e):
        """Draws sources from all populations based on the number choices and
        positions them.

        :param z_max: Maximum redshift for the sources.
        :param theta_e: Einstein radius of the deflector (used for
            scaling positions).
        :return: list of Source instances or None if any drawing fails.
        """
        all_sources = []

        # Iterate over each population and its corresponding number choices
        for pop, choices in zip(self._source_populations, self._number_choices):

            n_draw = random.choice(choices)

            for _ in range(n_draw):
                source = pop.draw_source()
                # If no source is available, return None
                if source is None:
                    return None
                all_sources.append(source)

        if not all_sources:
            return None

        # 3. Position all drawn sources
        # We position them collectively so they share the same spatial distribution logic
        total_sources = len(all_sources)

        for i, source in enumerate(all_sources):
            if self._clustering_mode == "ring":
                # Place roughly on the Einstein ring
                r = random.uniform(0.5 * theta_e, 2.5 * theta_e)
                # Distribute them evenly in angle (0 to 2pi) with jitter
                phi = (2 * np.pi * i / total_sources) + random.uniform(-0.3, 0.3)

                x_pos = r * np.cos(phi)
                y_pos = r * np.sin(phi)
            else:
                # Random placement in a box scaled by Einstein radius
                # Box limit is at least 1.5 arcsec or 3*theta_E
                box_limit = max(1.5, theta_e * 3.0)
                x_pos = random.uniform(-box_limit, box_limit)
                y_pos = random.uniform(-box_limit, box_limit)

            source.update_center(center_x=x_pos, center_y=y_pos)

        return all_sources

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

                # Step 2: Calculate Einstein radius for positioning
                theta_e_infinity = deflector.theta_e_infinity(cosmo=self.cosmo)
                test_area = self._test_area_factor * area_theta_e_infinity(
                    theta_e_infinity=theta_e_infinity
                )

                # Step 3: Draw and position sources from all populations
                sources = self.draw_sources(z_max, theta_e_infinity)
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
