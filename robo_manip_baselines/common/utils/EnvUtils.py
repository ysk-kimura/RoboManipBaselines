import pkgutil


def get_env_names():
    import robo_manip_baselines.envs.operation

    module_prefix = "Operation"

    env_names = [
        name[len(module_prefix) :]
        for _, name, _ in pkgutil.iter_modules(
            robo_manip_baselines.envs.operation.__path__
        )
        if name.startswith(module_prefix)
    ]

    return env_names
