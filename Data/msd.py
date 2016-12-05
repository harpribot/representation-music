import sqlite3
import operator
import numpy as np
import scipy as sp

TAGS_FILE = 'tags.txt'

def nontag_features():
    return ['hotttnesss',
            'duration',
            'key',
            'loudness',
            'year',
            'time signature',
            'tempo']

DEFAULT_VALUE = [0.0, 0.0, 0, 0.0, 0, 0, 0.0, [], []]
class MSFeatures:
    def __init__(self, values=DEFAULT_VALUE):
        """
        You may provide initial values for the features as a list in this order:

        1. Hotness
        2. Duration
        3. Key
        4. Loudness
        5. Year
        6. Time Signature
        7. Tempo
        8. Tags

        :param values: The values provided
        """
        self.hotness = values[0]
        self.duration = values[2]
        self.key = values[3]
        self.loudness = values[5]
        self.year = values[6]
        self.time_signature = values[7]
        self.tempo = values[8]
        self.tags = values[9]

        # For some reason, these are not populated in MSD.
        #self.danceability = values[1]
        #self.energy = values[4]

        # This is all the tags in a list. (A tag is a genre for this song)
        self.tagsList = self.tags.split('|')



    def feature(self, feature):
        '''
        Returns an integer or real number representing the queried feature variable.
        
        :param feature: Name of the feature.
        '''
        if feature == 'hotttnesss':
            return self.hotness
        elif feature == 'duration':
            return self.duration
        elif feature == 'key':
            return self.key
        elif feature == 'loudness':
            return self.loudness
        elif feature == 'year':
            return self.year
        elif feature == 'time signature':
            return self.time_signature
        elif feature == 'tempo':
            return self.tempo
        else:
            return self.isGenre(feature)
        

    def isGenre(self, genre):
        if genre in self.tagsList:
            return 1
        else:
            return 0


    def vector(self, features=[]):
        """
        Returns a feature vector of this song's features.
        :param tags:    List of tags to include as binary features.
        :return:        Vector of features.
        """
        vector = [self.feature(t) for t in tags]

        return vector 


class MillionSongFeatureDatabase:
    def __init__(self, db):
        """

        :param db:
        """
	# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	# db_path = os.path.join(BASE_DIR, db)
	conn = sqlite3.connect(db)
        self.db = conn.cursor()
        self._fstart = 2  # Feature start index

    def close(self):
        self.db.close()

    def get_features_by_msd(self, track_id):
        """
        Fetches features for a song by MSD ID.
        :param track_id: Million Songs Data set ID
        :return:
        """
        track_id = (track_id,)
        for row in self.db.execute('SELECT * FROM songs WHERE track_id=?', track_id):
            return MSFeatures(row[self._fstart:])

    def get_features_by_echo(self, echo_id):
        """
        Fetches features for a song by Echo Nest ID.
        :param echo_id: Echo Nest ID
        :return:
        """
        echo_id = (echo_id,)
        for row in self.db.execute('SELECT * FROM songs WHERE song_id=?', echo_id):
            return row[self._fstart:]

    def get_features_by_msd_list(self, track_ids):
        """

        :param track_ids:
        :return:
        """
        ids = set(track_ids)
        track_id_idx = 0
        for row in self.db.execute('SELECT * FROM songs WHERE ' + \
                                   'hotttness<>? AND ' + \
                                   # 'danceability<>? AND ' + \
                                   'duration<>? AND ' + \
                                   'key<>? AND ' + \
                                   # 'energy<>? AND ' + \
                                   'loudness<>? AND ' + \
                                   'year<>? AND ' + \
                                   'time_signature<>? AND ' + \
                                   'tempo<>?',
                                   (0.0, 0.0, 0,0.0,  0, 0, 0.0)):
            if row[track_id_idx] in ids:
                yield MSFeatures(row[self._fstart:])
        
    def get_songs_with_all_features(self):
        """

        :return: Returns a list of MSD IDs for all songs which have all features populated.
        """
        track_id_idx = 0
        for row in self.db.execute('SELECT * FROM songs WHERE ' + \
                                   'hotttness<>? AND ' + \
                                   # 'danceability<>? AND ' + \
                                   'duration<>? AND ' + \
                                   'key<>? AND ' + \
                                   # 'energy<>? AND ' + \
                                   'loudness<>? AND ' + \
                                   'year<>? AND ' + \
                                   'time_signature<>? AND ' + \
                                   'tempo<>?',
                                   (0.0, 0.0, 0, 0.0, 0, 0, 0.0)):
            yield row[track_id_idx]


class MillionSongLyricDatabase:
    def __init__(self, db, mappings):
        """
        :param db: Path to musiXmatch lyric database.
        :param mappings: Link to mappings file for lyric database. This should
                    be a single-line file with comma-separated word values
                    corresponding the 5,000 reported BOW words for the
                    musixmatch data set.
        """
       	# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	# db_path = os.path.join(BASE_DIR, db)
        conn = sqlite3.connect(db)
        self.db = conn.cursor()

        # The lyric database is built as a bag-of-words (BOW) for each song
        # over 5,000 carefully selected, discriminative, and common words
        # seen throughout the dataset. We map each of these words to an idx,
        # which is used to build a consistent BOW vector for any song.
        self.mappings = {}
        self._load_mappings(mappings)

    def close(self):
        self.db.close()

    def _load_mappings(self, mappings_file):
        """
        Loads BOW mappings from file.
        :param mappings_file:
        :return:
        """
        self.mappings = {}
        m = [line.rstrip('\n').split(',') for line in open(mappings_file)][0]
        for i, word in enumerate(m):
            self.mappings.update({word: i})

    def get_bow_by_msd_batch(self, track_ids):
        # Create an integer mapping for each desired track ID.
        track_mappings = {}
        count = 0
        for t in track_ids:
            track_mappings.update({ t : count })
            count += 1

        # Initialize a BOW matrix for all desired tracks.
        bow = [[0 for i in range(len(self.mappings))] for j in range(len(track_ids))]

        # Query database for all BOW and populate desired tracks.
        results = self.db.execute('SELECT track_id, word, count from lyrics')

        for r in results:
            track_id = r[0]
            word = r[1]
            count = r[2]
            try:
                if track_id in track_mappings:
                    bow[track_mappings[track_id]][self.mappings[word]] = count
            except:
                # Ignore key error. These occasionally arise from unicode nonsense.
                pass

        return bow

    def get_bow_by_msd(self, track_id):
        """
        Returns the BOW for a given MSD track ID, or an array of 0 counts if
        no track with the given ID is found in the database..
        :param track_id:
        :return:
        """
        try:
            results = self.db.execute('SELECT track_id, word, count from lyrics WHERE track_id=?', (track_id,))
            bow = [0] * len(self.mappings)
            for r in results:
                word = r[1]
                count = r[2]
                bow[self.mappings[word]] = count
        except:
            # Ignore key error. These occasionally arise from unicode nonsense.
            bow = [0] * len(self.mappings)
        return bow

    def get_songs_with_lyrics(self):
        """
        Returns a list of all songs in the lyric database by MSD track ID.
        """
        try:
            for row in self.db.execute('SELECT track_id from lyrics'):
                yield row[0]
        except:
            pass


class MillionSongDataset:
    def __init__(self, features, lyrics, lyric_mappings, tracks):
        """

        :param features:
        :param lyrics:
        :param lyric_mappings:
        :param tracks:
        """
        # Initialize lyric and feature databases.
        self._fdb = MillionSongFeatureDatabase(features)
        self._ldb = MillionSongLyricDatabase(lyrics, lyric_mappings)

        # Load track.
        self.tracks = []
        for line in [line.rstrip() for line in open(tracks)]:
            self.tracks.append(line)

        self.train = []
        self.validate = []
        self.test = []

    def close(self):
        self._fdb.close()
        self._ldb.close()
        

    def generate_track_list(self):
        """
        Generates a list of songs with both full features and lyrics and
        saves to a target file.
        :return: None
        """
        with_lyrics = set(self._ldb.get_songs_with_lyrics())
        with_features = set(self._fdb.get_songs_with_all_features())
        with_both = with_lyrics.intersection(with_features)

        with open('tracks.txt', 'wb') as f:
            for t in with_both:
                f.write('%s\n' % t)

    def load_track_list(self, f):
        """

        :param f:
        :return: None
        """
        for line in [line.rstrip for line in open(f)]:
            self.tracks += [line]

    def get_features(self, track_ids):
        """

        :param track_ids:
        :return:
        """
        return self._fdb.get_features_by_msd_list(track_ids)

    '''
    def get_bow(self, track_ids):
        count = 0
        for t in track_ids:
            count += 1
            print count
            yield self._ldb.get_bow_by_msd(t)
    '''

    def get_bow(self, track_ids):
        """

        :param track_ids:
        :return:
        """
        return self._ldb.get_bow_by_msd_batch(track_ids)

    def generate_split(self, train, validate, test, total=147000):
        """
        Generates a train/validate/test split of the data.
        :param train: [0,1] portion used for training
        :param validate: [0,1] portion used for validate
        :param test: [0,1] portion used for test
        :param total: total number of examples used from all 3.
        :return: None
        """
        assert(train + test + validate == 1.0)

        total = min(total, len(self.tracks))
        end_train = int(total*train)
        end_validate = int(total*validate) + end_train
        self.train = self.tracks[:end_train]
        self.validate = self.tracks[end_train:end_validate]
        self.test = self.tracks[end_validate:total]

def load_tags():
    '''
    Loads music brainz tags from specified tag file and returns them as a list.
    '''
    try:
        return [line.rstrip() for line in open(TAGS_FILE)]
    except:
        print ("Failed to load music brainz tags from file.")
        return []
        

def correlated_features(data, target_feature, k):
    '''
    Given a tag, returns the k most highly correlated features
    with that tag over the input.
    
    :param data:      Input data in form of nparray of MSFeatures objects.
    :param feature:   String of feature to match.
    :param k:         Number of results.
    :return:          String names of k most highly correlated features.
    '''
    # Load all tags from file.
    tags = load_tags()

    # Make sure the feature is valid.
    if target_feature not in tags:
        print ("Feature does not exist. Cannot find correlated features.")
        return np.array([])

    # Remove target feature from list of tags.
    tags.remove(target_feature)

    # Extract feature array over input data.
    target_vector = [song.feature(target_feature) for song in data]
    
    # Find features which are most highly correlated.
    all_features = nontag_features()
    all_features.extend(tags)
    corrs = {}
    count = 0
    for f in all_features:
        f_vector = [song.feature(f) for song in data]
        corr = np.corrcoef(np.array([target_vector, f_vector], np.float))[0][1]
        if np.isnan(corr):
            corr = 0.0
        corrs.update({ f : corr })
        count += 1
    sorted_corr = sorted(corrs.items(), key=operator.itemgetter(1))
    for f in list(reversed(sorted_corr)):
        print f

        
        


if __name__ == '__main__':
    features       = '../Data/msongs.db'
    lyrics         = '../Data/mxm_dataset.db'
    lyric_mappings = '../Data/bow.txt'
    tracks         = '../Data/tracks.txt'
    db = MillionSongDataset(features, lyrics, lyric_mappings, tracks)
    db.generate_split(0.05, 0.15, 0.80, 145000)

    features = [t for t in db.get_features(db.train + db.validate + db.test)]
    #bow = [bow for bow in db.get_bow(db.train)]
    
    correlated_features(features, 'pop', 5)


'''    
# Sample Usage - Load the BOW and features for a single track.
track = db.tracks[0]
print db.get_features(track).vector()
print db.get_bow(track)
'''


# Sample Usage - Make a train/validate/test split and acquire the features for each item in train.
'''
db.generate_split(0.75, 0.10, 0.15, 5000)
features = [t.vector() for t in db.get_features(db.train)]
bow = [bow for bow in db.get_bow(db.train)]
'''
