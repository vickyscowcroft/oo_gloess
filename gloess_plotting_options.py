import numpy as np

glo_cols = {'U' : 'Violet', 'B' : 'MediumSlateBlue', 'V' : 'DodgerBlue', 'R': 'Turquoise', 
                    'I': 'LawnGreen', 'J': 'Gold', 'H': 'DarkOrange', 'Ks' :'HotPink', 'K': 'HotPink', 'IRAC1' : 'MediumVioletRed', 
                    'IRAC2' : 'DeepPink', 'IRAC3' :'HotPink', 'IRAC4' : 'PeachPuff', 'G': 'Green', 'BP': 'Blue',
                    'RP': 'Red', 'W1' : 'MediumVioletRed', 
                    'W2' : 'DeepPink', 'W3' :'HotPink', 'W4' : 'PeachPuff', 'TESS': 'silver'}
glo_offs = {'U' : 3, 'B' : 1.5, 'V' : 1.2, 'R': 0.7, 'I': 0.2, 'J': 0, 'H': -0.4, 'Ks' :-0.8, 'K': -0.8,
                    'IRAC1' : -1.4, 'IRAC2' : -1.8, 'IRAC3' :-2.2, 'IRAC4' : -2.6, 'G': 1, 'BP': 1, 'RP':1, 
                    'W1' : -1.4, 'W2' : -1.8, 'W3' :-2.2, 'W4' : -2.6}
#glo_offs = {'U' : 0, 'B' : 0, 'V' : 0, 'R': 0, 'I': 0, 'J': 0, 'H': 0, 'Ks' :0, 'K': 0,
#                    'W1' : 0, 'W2' : 0, 'W3' :0, 'W4' : 0, 'TESS': -0.5}
#                    'IRAC1' : 0, 'IRAC2' : 0, 'IRAC3' :0, 'IRAC4' : 0, 'G': 0, 'BP': 0, 'RP':0, 
plot_symbols = {'U' : 'o', 'B' : 'o', 'V' : 'o', 'R': 'o', 'I': 'o', 'J': 'o', 'H': 'o', 'Ks' :'o', 'K': 'o',
                        'IRAC1' : 'o', 'IRAC2' : 'o', 'IRAC3' :'o', 'IRAC4' : 'o', 'G': 'o', 'BP': 'o', 'RP': 'o', 
                        'W1' : 'x', 'W2' : 'x', 'W3' :'x', 'W4' : 'x', 'TESS': 'o'}

def make_offset_label(band, offset):
    if offset==0:
        label = f'{band}'
    elif offset < 0:
        label = f'{band} $-$ {np.abs(offset)}'
    else:
        label = f'{band} + {offset}'
    return(label)
