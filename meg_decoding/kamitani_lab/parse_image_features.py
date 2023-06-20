import h5py
import numpy as np


def parse_h5(file, extract_layer='cnn5'):
    """
    data['dataSet'] shape
    ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8',
    'hmax1', 'hmax2', 'hmax3',
    'gist', 'sift', 'ImageID', 'CatID', 'FeatureType', 'UnitIndex']

    >> data['dataSet']
    <HDF5 dataset "dataSet": shape (16622, 13027), type "<f8">
    """
    with h5py.File(file, 'r') as f:
        for i, l in enumerate(f['metaData']['key']):
            layer_name = l.decode('utf-8')
            if layer_name == extract_layer:
                layer_correspond_ids = np.where(~np.isnan(f['metaData']['value'][i])) # [0, 13027-1]から対応するものだけ取り出す
        features = f['dataSet'][:, layer_correspond_ids]