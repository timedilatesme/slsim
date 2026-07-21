class LOSIndividual(object):
    """Class to store the quantities of an individual line of sight."""

    def __init__(self, kappa=None, gamma=None, gamma_1=None, gamma_2=None):
        """

        :param gamma: [gamma1, gamma2] (takes these values if present)
        :type gamma: list of floats
        :param gamma_1: gamma1 component
        :param gamma_2: gamma2 component
        :param kappa: convergence (takes this values if present)
        :type kappa: float
        """
        if kappa is None:
            kappa = 0
        if gamma_1 is None or gamma_2 is None:
            if gamma is None:
                gamma = [0, 0]
        else:
            gamma = [gamma_1, gamma_2]
        self._kappa = kappa
        self._gamma = gamma

    @property
    def convergence(self):
        """Line of sight convergence.

        :return: kappa
        """
        return self._kappa

    @property
    def shear(self):
        """Line of sight shear.

        :return: gamma1, gamma2
        """
        return self._gamma[0], self._gamma[1]
