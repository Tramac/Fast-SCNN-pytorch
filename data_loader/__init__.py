from .cityscapes import CitySegmentation

datasets = {
    'citys': CitySegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
