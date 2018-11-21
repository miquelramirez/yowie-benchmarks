"""
    Utilities for the environments
"""

from tarski.syntax import *
from tarski.syntax.arithmetic import *
from tarski.syntax.arithmetic.special import *
from tarski.syntax.arithmetic.random import *

def huber(x, p):
    return sqrt(x*x + p*p) - p
