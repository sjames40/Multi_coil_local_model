import os
import xarray as xr
import contextlib
from pathlib import Path

from src.constants import DATA_DIR

MISSING_NAME_VAL = 'name_missing'


@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
data_names = {
    'mono': 'Mono-energetic_1pt5MeV_1material_5densities_Collimator_MassConservation_v2.zarr',
    'Brem': 'Brem_1material_5densities_Collimator_MassConservation.zarr',
    'multi': '6MeV_Brem_5materials_noAir_NoCollimator_NoMassConservation_ArealMassLE200.zarr',
    'Brem_2x': 'Brem_1material_5densities_Collimator_MassConservation_Density2x.zarr',
    'Brem_3x': 'Brem_1material_5densities_Collimator_MassConservation_Density3x.zarr',
    'con_estimate':'out_local_conv.zarr',
    'con_direct':'out_simple.zarr',
    'conv_estimate':'out_mult1local_conv.zarr',
    'mono_conv':'out_mono1local_conv.zarr',
    'Brem_conv':'out_bremlocal_conv.zarr',
    'multi_conv':'out_conv_testlocal_two_layer_conv.zarr',
    'multi_conv2':'out_conv_1local_two_layer_conv.zarr',
    'multi_conv3':'out_conv_test_correctlocal_two_layer_conv.zarr',
    'multi_3layer':'out_conv_test_3layer_20neiglocal_two_layer_conv.zarr',
    'multi_2layer':'out_conv_test_2layer_20neiglocal_two_layer_conv.zarr',

    }

def resultname_to_type(name):
    return os.path.basename(name).split('-')[0]


def resultname_to_dataset(name):
    return '-'.join(os.path.basename(name).split('-')[1:-1])


def resultname_to_method(name):
    return os.path.basename(name).split('-')[-2]


def save_array(x, filename, *args, **kwargs):
    """
    x - DataArray
    """

    # add .zarr extension if needed
    if not filename.endswith('.zarr'):
        filename = filename + '.zarr'

    # add name if needed
    if x.name is None:
        x.name = MISSING_NAME_VAL

    # convert to DataSet and save
    # (DataArray objects can't save directly right now)
    x.to_dataset().to_zarr(filename, *args, **kwargs)

def get_groups(folder):
    return [ f.name for f in os.scandir(folder) if f.is_dir() ]

def load_array(filename, group=None, into_memory=True):
    """
    """
    # add zarr extension if needed
    if not (filename.endswith('.zarr') or filename.endswith('.zarr/')):
        filename = filename + '.zarr'

    try:
        x = xr.open_zarr(filename, group)
    except ValueError as e:
        msg = f'The zarr file at {filename} does exist or does not contain {group}'
        raise ValueError(msg) from e

    x = x[list(x.data_vars)[0]]  # turn dataset into dataarray


    if into_memory:
        x.load()  # reads from disk into memory
        x.close()

    # remove placeholder name
    if x.name == MISSING_NAME_VAL:
        x.name = None

    return x

def prep1(data_short_name, images=True, small=False):
    """
    Usage:
    rho, D, S, T, test, train = ds.data.prep('mono')
    D.sel(test)  # to get testing Ds
    S.sel(train)  # to get training Ss
    """

    if small is True:
        raise ValueError('using small is deprecated')

    test_size = 10
    max_train_size = 1000

    #max_total_ims = test_size + max_train_size

    data_full_name = data_names[data_short_name]
    dataset_path = os.path.join(DATA_DIR, data_full_name)
    D = load_array(dataset_path, 'direct_image_estimate', into_memory=False)
    return D
data_name ='con_estimate'

D = prep1(data_name)











def prep(data_short_name, images=True, small=False):
    """
    Usage:
    rho, D, S, T, test, train = ds.data.prep('mono')
    D.sel(test)  # to get testing Ds
    S.sel(train)  # to get training Ss
    """

    if small is True:
        raise ValueError('using small is deprecated')

    test_size = 10
    max_train_size = 1000

    max_total_ims = test_size + max_train_size

    data_full_name = data_names[data_short_name]

    dataset_path = os.path.join(DATA_DIR, data_full_name)

    if images and small:
        rho = load_array(dataset_path, 'density_small_image', into_memory=False)
        D = load_array(dataset_path, 'direct_small_image', into_memory=False)
        S = load_array(dataset_path, 'total_scatter_small_image', into_memory=False)
    elif images:
        rho = load_array(dataset_path, 'density_image', into_memory=False)
        D = load_array(dataset_path, 'direct_image', into_memory=False)
        S = load_array(dataset_path, 'total_scatter_image', into_memory=False)
    else:  # profiles
        rho = load_array(dataset_path, 'density_profile', into_memory=False)
        D = load_array(dataset_path, 'direct_profile', into_memory=False)
        S = load_array(dataset_path, 'total_scatter_profile', into_memory=False)

    common_inds = list(set(D.ID.data).intersection(set(S.ID.data)))

    total_ims = min(len(common_inds), max_total_ims)
    train_size = total_ims - test_size

    if data_short_name is 'multi':  # avoid bad test image ID=4
        test = dict(ID=common_inds[train_size:total_ims])
        train = dict(ID=common_inds[:train_size])
    else:
        test = dict(ID=common_inds[:test_size])
        train = dict(ID=common_inds[test_size:total_ims])
    test_and_train = dict(ID=common_inds[:total_ims])

    rho = rho.sel(test_and_train)  # can't easily do this in a loop
    D = D.sel(test_and_train)
    S = S.sel(test_and_train)

    T = D + S  # only adds corresponding IDs
    T.name = 'total_transmission'

    for X in D, S, rho, T:
        X.attrs['data_name'] = data_short_name
        X.attrs['data_full_name'] = data_full_name
        X.load()

    return rho, D, S, T, test, train
