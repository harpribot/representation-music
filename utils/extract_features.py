import os
import glob
import sys
import sqlite3
import hdf5_getters
import numpy as np


# Features of interest.
features = ['track_id',
            'song_id',
            'hotttnesss',
            'danceability',
            'duration',
            'key',
            'energy',
            'loudness',
            'year',
            'time_signature',
            'tempo',
            'tags']


def create_table(c):
    '''
    Creates a new table for songs from scratch.
    '''
    c.execute('''CREATE TABLE songs
                 (track_id, song_id, hotttness, danceability, duration, key, energy, loudness, year, time_signature, tempo, tags)''')

def write_features_to_db(c, features):
    '''
    Writes features to database. Assumes the correct column ordering.
    '''
    #print features
    c.execute('INSERT INTO songs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', features)

def extract_features(filename):
    h5 = hdf5_getters.open_h5_file_read(filename)
    f = [None] * len(features)
    f[features.index('track_id')] = hdf5_getters.get_track_id(h5, 0).item()
    f[features.index('song_id')] = hdf5_getters.get_song_id(h5, 0).item()
    f[features.index('hotttnesss')] = hdf5_getters.get_artist_hotttnesss(h5, 0).item()
    f[features.index('danceability')] = hdf5_getters.get_danceability(h5, 0).item()
    f[features.index('duration')] = hdf5_getters.get_duration(h5, 0).item()
    f[features.index('key')] = hdf5_getters.get_key(h5, 0).item()
    f[features.index('energy')] = hdf5_getters.get_energy(h5, 0).item()
    f[features.index('loudness')] = hdf5_getters.get_loudness(h5, 0).item()
    f[features.index('year')] = hdf5_getters.get_year(h5, 0).item()
    f[features.index('time_signature')] = hdf5_getters.get_time_signature(h5, 0).item()
    f[features.index('tempo')] = hdf5_getters.get_tempo(h5, 0).item()
    tags = ''
    for tag in hdf5_getters.get_artist_terms(h5):
        tags += ('%s|' % tag)
    # Remove trailing pipe.
    tags = tags[:len(tags) - 1] 
    f[features.index('tags')] = tags
    h5.close()
    return f

# Connect to database.
conn = sqlite3.connect('msongs.db')
c = conn.cursor()
create_table(c)

directory = '/home/tyler/Projects/IR/MultiTaskDeepRepresentationLearning/data'

files = []
file_count = 0
count = 0
#limit = 10
limit = sys.maxint

def get_all_files(basedir,ext='.h5') :
    """
    From a root directory, go through all subdirectories
    and find all files with the given extension.
    Return all absolute paths in a list.
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files :
            allfiles.append( os.path.abspath(f) )
    return allfiles


files = get_all_files(directory)
for file in files:
    print ("Extracting (%d): %s" % (count, ""))
    try:
        extracted = extract_features(file)
        write_features_to_db(c, extracted)
        count += 1
    except:
        print ("Failed: %s" % file)
    
        


# Print db.
print "Printing database..."
for row in c.execute('SELECT * FROM songs ORDER BY track_id'):
    print row

print file_count
print count
conn.commit()

