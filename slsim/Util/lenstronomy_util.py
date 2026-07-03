from jaxtronomy.LensModel.profile_list_base import _JAXXED_MODELS

def jax_usage(use_jax, lens_mass_model_list):
    """
    get all the JAX profiles if enabled from jaxtronomy

    :param use_jax: if True aims to use all the available jaxtronomy profiles
    :type use_jax: bool
    :param lens_mass_model_list:
    :return:
    """

    if use_jax is True:
        _use_jax = []
        for profile in lens_mass_model_list:
            if profile in _JAXXED_MODELS:
                _use_jax.append(True)
            else:
                _use_jax.append(False)
    else:
        _use_jax = False
    return _use_jax