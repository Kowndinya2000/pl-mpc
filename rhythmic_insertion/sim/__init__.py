"""Namespaced IsaacGymEnvs components used by rhythmic insertion."""
from omegaconf import OmegaConf

def register_resolvers():
    resolvers = {
        "eq": lambda x, y: str(x).lower() == str(y).lower(),
        "contains": lambda x, y: str(x).lower() in str(y).lower(),
        "if": lambda pred, a, b: a if pred else b,
        "resolve_default": lambda default, arg: default if arg == "" else arg,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
