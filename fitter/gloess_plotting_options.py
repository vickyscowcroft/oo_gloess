import numpy as np
from matplotlib.pyplot import cm
import pandas as pd
from . import DATADIR

standard_filters = DATADIR + '/standard_filter_wavelengths.csv'
glo_plots_df = pd.read_csv(standard_filters)

glo_plots_df.sort_values('wavelength_um', inplace=True)
glo_plots_df.reset_index(drop=True, inplace=True)

glo_plot_order = pd.Series(glo_plots_df.index.values,index=glo_plots_df.band).to_dict() 

glo_cols = pd.Series(glo_plots_df.plotcolor.values,index=glo_plots_df.band).to_dict()
plot_symbols = pd.Series(glo_plots_df.plotsymbol.values,index=glo_plots_df.band).to_dict()
glo_offs = pd.Series(glo_plots_df.offset_calc.values,index=glo_plots_df.band).to_dict() 
glo_labels = pd.Series(glo_plots_df.label.values,index=glo_plots_df.band).to_dict() 
glo_mag_axis = pd.Series(glo_plots_df.ismags.values,index=glo_plots_df.band).to_dict() 
glo_bands = glo_plots_df.band.values

glo_standard_filters = pd.Series(glo_plots_df.is_JC_IR.values,index=glo_plots_df.band).to_dict() 


def extra_bands(bands):
    color = iter(cm.rainbow(np.linspace(0, 1, 10)))
    n_extra = len(glo_plots_df)
    for i in range(len(bands)):
        if bands[i] not in glo_bands:
            n_extra = n_extra + 1
            c = next(color)
            glo_cols[bands[i]] = c
            glo_offs[bands[i]] = 0.0
            plot_symbols[bands[i]] = '*'
            glo_standard_filters[bands[i]] = False
            glo_labels[bands[i]] = bands[i]
            glo_mag_axis[bands[i]] = True
            glo_plot_order[bands[i]] = n_extra
    return(glo_cols, glo_offs, plot_symbols, glo_standard_filters, glo_plot_order, glo_labels,glo_mag_axis)

def make_offset_label(band, offset):
    if offset==0:
        label = f'{band}'
    elif offset < 0:
        label = f'{band} $-$ {np.abs(offset):.1f}'
    else:
        label = f'{band} + {offset:.1f}'
    return(label)

