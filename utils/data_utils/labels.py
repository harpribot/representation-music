from enum import Enum

# Previous static implementation. Does not support IDs for many genre tag features.
'''
class Labels(Enum):
    hotness = '0'
    duration = '1'
    key = '2'
    loudness = '3'
    year = '4'
    time_signature = '5'
    tempo = '6'
    tags = '7'
'''

class Labels(object):
    """
    Given a list of features, assigns each an ID starting at zero.
    """

    def __init__(self, labels):
        """

        :param labels: Collection of labels to create IDs for.
        """
        self.count = 0
        self.ids = {}
        for l in labels:
            self.ids.update({ l : self.count })
            self.count += 1

    def get(self, label):
        """

        :return: The ID associated with a label, or None if there is not one.
        """
        try:
            return self.ids[label]
        except:
            return None
