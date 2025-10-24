from os.path import dirname, join as joinpath
from pathlib import Path

thispath = Path(__file__)
dirpath = thispath.parent.parent

DATADIR = joinpath(dirpath, 'data')

from . import gloess_fitter
from . import gloess_plotting_options as gf_plot


