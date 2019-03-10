from __future__ import print_function

import sys


def debug(expression):
    frame = sys._getframe(1)

    print(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))
