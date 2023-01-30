import gloesspy3 as gf
import pandas as pd
import numpy as np

def clean_old_gloess_file(filename):
    df, period, smoothing = gf.old_gloess_to_df(filename, ref_col=True)
    bad_refs = []
    bad_ref_files = ['bad_references_list.txt', 'johnson_system_references.txt', 'not_irac_references.txt']
    for file in bad_ref_files:
        with open('/Users/vs522/Dropbox/Python/oo_gloess/'+file) as fn:
            for line in fn:
                line = line.strip()
                bad_refs.append(line)
    df = df[~df['Reference'].isin(bad_refs)]
    df['MJD'].replace(-99.99, np.NaN, inplace=True)
    df.dropna(subset=['MJD'], inplace=True)
    return(df)



