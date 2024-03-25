from __future__ import absolute_import
from .pkginfo import __splash__

import os
if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
    # then process is launched in distributed mode.
    # don't print splash screen
    pass
else:
    print(__splash__)

from . import *

