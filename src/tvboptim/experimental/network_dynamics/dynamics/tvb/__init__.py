"""Neural mass models."""

from .coombes_byrne import CoombesByrne2D
from .epileptor import Epileptor
from .generic_2d_oscillator import Generic2dOscillator
from .jansen_rit import JansenRit
from .kuramoto import Kuramoto
from .larter_breakspear import LarterBreakspear
from .linear import Linear
from .montbrio_pazo_roxin import MontbrioPazoRoxin
from .sup_hopf import SupHopf
from .wilson_cowan import WilsonCowan
from .wong_wang import ReducedWongWang
from .wong_wang_exc_inh import WongWangExcInh

__all__ = [
    "CoombesByrne2D",
    "Epileptor",
    "Generic2dOscillator",
    "JansenRit",
    "Kuramoto",
    "LarterBreakspear",
    "Linear",
    "MontbrioPazoRoxin",
    "SupHopf",
    "WilsonCowan",
    "ReducedWongWang",
    "WongWangExcInh",
]
