import tarski
from tarski.theories import Theory
from tarski.syntax import *

def zermelo():
    lang = tarski.language('zermelo', [Theory.EQUALITY, Theory.ARITHMETIC])

    x = lang.function('x', lang.Object, lang.Real)
    y = lang.function('y', lang.Object, lang.Real)
    wx = lang.function('wx', lang.Real)
    wy = lang.function('wy', lang.Real)
    ux = lang.function('ux', lang.Object, lang.Real)
    uy = lang.function('uy', lang.Object, lang.Real)

    s = lang.constant('s', lang.Object)

    return lang, x, y, wx, wy, ux, uy, s

def uav_teams():
    lang = tarski.language('uav_teaming', [Theory.EQUALITY, Theory.ARITHMETIC, Theory.SPECIAL])
    x = lang.function('x', lang.Object, lang.Real)
    y = lang.function('y', lang.Object, lang.Real) # coordinates
    vx = lang.function('v_x', lang.Object, lang.Real)
    vy = lang.function('v_y', lang.Object, lang.Real) # velocity vector
    b = lang.function('b', lang.Object, lang.Real)
    wx = lang.function('w_x', lang.Real)
    wy = lang.function('w_y', lang.Real)
    r = lang.function('r', lang.Object, lang.Real) # radius

    s = lang.constant('s', lang.Object)

    return lang, x, y, b, vx, vy, wx, wy, r, s
