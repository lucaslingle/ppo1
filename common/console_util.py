"""
Console util. Ported necessary components from
https://github.com/openai/baselines/blob/master/baselines/common/console_util.py
"""

import numpy as np


def fmt_row(width, row, header=False):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/console_util.py#L12
    """
    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-"*len(out)
    return


def fmt_item(x, l):
    """Reference:
    https://github.com/openai/baselines/blob/master/baselines/common/console_util.py#L17
    """
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, (float, np.float32, np.float64)):
        v = abs(x)
        if (v < 1e-4 or v > 1e+4) and v > 0:
            rep = "%7.2e" % x
        else:
            rep = "%7.5f" % x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep
