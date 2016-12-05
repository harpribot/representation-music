from utils.network_utils.params import LossTypes


class Coupled(object):
    tasks = {
        'tightly_coupled': {'pop': LossTypes.cross_entropy,
                            'pop rock': LossTypes.cross_entropy,
                            'ballad': LossTypes.cross_entropy},
        'loosely_coupled': {'pop': LossTypes.cross_entropy,
                            'loudness': LossTypes.mse,
                            'year': LossTypes.mse}
    }


class Individual(object):
    tasks = {
        'target': {'pop': LossTypes.cross_entropy},
        'dependent_1': {'pop rock': LossTypes.cross_entropy},
        'dependent_2': {'ballad': LossTypes.cross_entropy},
        'dependent_3': {'loudness': LossTypes.mse},
        'dependent_4': {'year': LossTypes.mse}
    }
