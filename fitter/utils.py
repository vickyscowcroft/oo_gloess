import pyvo as vo
from astroquery.gaia import Gaia



def get_gaia_source_id(name, gaia_release='DR3'):
    """ gets the gaia source id for an object from simbad
        defaults to DR3 source id (identical to EDR3)
    """
    tap_service = vo.dal.TAPService("http://simbad.cds.unistra.fr/simbad/sim-tap")
    query_string = f"SELECT id2.id FROM ident AS id1 JOIN ident AS id2 USING(oidref) WHERE id1.id = \
        '{name}' and id2.id like 'Gaia {gaia_release}%';"
    tap_results = tap_service.search(query_string)
    if len(tap_results) > 0:
        value = tap_results['id', 0]
        source_id = value.split(' ')[2]
    else:
        source_id = np.nan
    return(source_id)

def get_gaia_period(source_id, var_type='cep', period_col='pf'):
    if var_type=='cep':
        query = f"select source_id, {period_col} from gaiadr3.vari_cepheid where source_id in ({source_id})"
    elif var_type=='rrl':
        query = f"select source_id, {period_col} from gaiadr3.vari_rrlyrae where source_id in ({source_id})"
    job     = Gaia.launch_job_async(query)
    period = job.get_results()[period_col][0]
    return(period)
