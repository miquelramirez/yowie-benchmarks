import tarski
from tarski.theories import Theory
from tarski.syntax import *
from yowie.fstrips import *
import yowie.fstrips.geometry as fsg


def uav_fly(lang, durations, where_to=None):

    x, y = lang.get('x'), lang.get('y')
    vx, vy = lang.get('v_x'), lang.get('v_y')
    b = lang.get('b')
    s = lang.get('s')

    dynamics = [
        (x(s), vx(s)),
        (y(s), vy(s)),
        (b(s), -0.1*(square(vx(s))+square(vy(s))))
    ]

    # activities
    if where_to is None:
        fly = HybridActivity(name='fly',
            dur=durations,
            # inside AO
            invariant=[],
            eff_cont= dynamics
        )
    else:
        fly = HybridActivity(name='fly',
            dur=durations,
            # inside AO
            invariant=[],
            eff_cont= dynamics,
            pre_end=[where_to]
        )


    return fly

def uav_visit_poi(lang, within_poi, how_long=20):

    keep_station_on_poi = HybridActivity(
        name='keep_station',
        dur=(how_long, how_long),
        pre_start=[within_poi],
        invariant=[within_poi],
        pre_end=[within_poi]
    )
    return keep_station_on_poi
