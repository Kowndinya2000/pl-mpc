try:
    import isaacgym
except Exception:
    pass

from .tdmpc_square import TDMPC2

__all__ = ["TDMPC2"]
