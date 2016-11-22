import sqlite3
import os
import sys


class MSFeatures:

    def __init__(self, values=None):
        '''
        You may provide initial values for the features as a list in this order:
        
        1. Hotttnesss
        2. Duration
        3. Key
        4. Loudness
        5. Year
        6. Time Signature
        7. Tempo
        8. Tags
        '''
        if values:
            self.hotttnesss = values[0]
            self.duration = values[1]
            self.key = values[2]
            self.loudness = values[3]
            self.year = values[4]
            self.time_signature = values[5]
            self.tempo = values[6]
            self.tags = values[7]
        else:
            self.hotttnesss = 0.0
            self.duration = 0.0
            self.key = 0
            self.loudness = 0.0
            self.year = 0
            self.time_signature = 0
            self.tempo = 0.0
            self.tags = []

    def vector(self):
        return [self.hotttnesss,
                self.duration,
                self.key,
                self.loudness,
                self.year,
                self.time_signature,
                self.tempo]


class MillionSongFeatureDatabase:

    def __init__(self, db):
        conn = sqlite3.connect(db)
        self.db = conn.cursor()
        self._fstart = 2 # Feature start index

    def close(self):
        self.db.close()

    def get_features_by_msd(self, track_id):
        '''
        Fetches features for a song by MSD ID.
        '''
        track_id = (track_id,)
        for row in self.db.execute('SELECT * FROM songs WHERE track_id=?', track_id):
            return MSFeatures(row[self._fstart:])

    def get_features_by_echo(self, echo_id):
        '''
        Fetches features for a song by Echo Nest ID.
        '''
        echo_id = (echo_id,)
        for row in self.db.execute('SELECT * FROM songs WHERE song_id=?', echo_id):
            return row[self._fstart:]

    def get_features_by_msd_list(self, track_ids):
        ids = set(track_ids)
        track_id_idx = 0
        for row in self.db.execute('SELECT * FROM songs WHERE ' + \
                                   'hotttness<>? AND ' + \
                                   #'danceability<>? AND ' + \
                                   'duration<>? AND ' + \
                                   'key<>? AND ' + \
                                   #'energy<>? AND ' + \
                                   'loudness<>? AND ' + \
                                   'year<>? AND ' + \
                                   'time_signature<>? AND ' + \
                                   'tempo<>?',
                                   (0.0, 0.0, 0,0.0,  0, 0, 0.0)):
            if row[track_id_idx] in ids:
                yield MSFeatures(row[self._fstart:])
        
    def get_songs_with_all_features(self):
        '''
        Returns a list of MSD IDs for all songs which have all features populated.
        '''
        track_id_idx = 0
        for row in self.db.execute('SELECT * FROM songs WHERE ' + \
                                   'hotttness<>? AND ' + \
                                   #'danceability<>? AND ' + \
                                   'duration<>? AND ' + \
                                   'key<>? AND ' + \
                                   #'energy<>? AND ' + \
                                   'loudness<>? AND ' + \
                                   'year<>? AND ' + \
                                   'time_signature<>? AND ' + \
                                   'tempo<>?',
                                   (0.0, 0.0, 0,0.0,  0, 0, 0.0)):
            yield row[track_id_idx]

            


            
class MillionSongLyricDatabase:
    
    def __init__(self, db, mappings):
        '''
        db       -- Path to musiXmatch lyric database.
        mappings -- Link to mappings file for lyric database. This should
                    be a single-line file with comma-separated word values
                    corresponding the 5,000 reported BOW words for the
                    musiXmatch dataset.
        '''
        conn = sqlite3.connect(db)
        self.db = conn.cursor()

        # The lyric database is built as a bag-of-words (BOW) for each song
        # over 5,000 carefully selected, discriminative, and common words
        # seen throughout the dataset. We map each of these words to an idx,
        # which is used to build a consistent BOW vector for any song.
        self.mappings = {}
        self._load_mappings(mappings)

    def _load_mappings(self, mappings_file):
        '''
        Loads BOW mappings from file. 
        '''
        self.mappings = {}
        m = [line.rstrip('\n').split(',') for line in open(mappings_file)][0]
        for i, word in enumerate(m):
            self.mappings.update({ word : i })

    def get_bow_by_msd(self, track_id):
        '''
        Returns the BOW for a given MSD track ID, or an array of 0 counts if
        no track with the given ID is found in the database..
        '''
        try:
            results = self.db.execute('SELECT track_id, word, count from lyrics WHERE track_id=?', (track_id,))
            bow = [0] * len(self.mappings)
            for r in results:
                word  = r[1]
                count = r[2]
                bow[self.mappings[word]] = count
        except:
            bow = [0] * len(self.mappings)
        return bow
            
            
    def get_songs_with_lyrics(self):
        '''
        Returns a list of all songs in the lyric database by MSD track ID.
        '''
        try:
            for row in self.db.execute('SELECT track_id from lyrics'):
                yield row[0]
        except:
            pass




            
class MillionSongDataset:

    def __init__(self, features, lyrics, lyric_mappings, tracks):
        # Initialize lyric and feature databases.
        self._fdb = MillionSongFeatureDatabase(features)
        self._ldb = MillionSongLyricDatabase(lyrics, lyric_mappings)

        # Load track.
        self.tracks = []
        for line in [line.rstrip() for line in open(tracks)]:
            self.tracks.append(line)

        self.train    = []
        self.validate = []
        self.test     = []

    def generate_track_list(self):
        '''
        Generates a list of songs with both full features and lyrics and
        saves to a target file.
        '''
        with_lyrics   = set(self._ldb.get_songs_with_lyrics())
        with_features = set(self._fdb.get_songs_with_all_features())
        with_both     = with_lyrics.intersection(with_features)
        with open('tracks.txt', 'wb') as f:
            for t in with_both:
                f.write('%s\n' % t)

    def load_track_list(self, f):
        for line in [line.rstrip for line in open(f)]:
            self.tracks += [line]

    def get_features(self, track_ids):
        return self._fdb.get_features_by_msd_list(track_ids)

    def get_bow(self, track_ids):
        for t in track_ids:
            yield self._ldb.get_bow_by_msd(t)

    def generate_split(self, train, validate, test, total=147000):
        '''
        Generates a train/validate/test split of the data.
        train = [0,1] portion used for training
        test = [0,1] portion used for validate
        validate = [0,1] portion used for test
        total = total number of examples used from all 3.
        '''
        assert(train + test + validate == 1.0)

        total = float(min(total, len(self.tracks)))
        end_train = int(total*train)
        end_validate = int(total*validate)
        self.train    = self.tracks[:end_train]
        self.validate = self.tracks[end_train:end_validate]
        self.test     = self.tracks[end_validate:]





features       = '../Data/msongs.db'
lyrics         = '../Data/mxm_dataset.db'
lyric_mappings = '../Data/bow.txt'
tracks         = '../Data/tracks.txt'
db = MillionSongDataset(features, lyrics, lyric_mappings, tracks)


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




