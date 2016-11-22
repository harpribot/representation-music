import sqlite3
import sys

def main():
  
	# File to write track_id values to.
	target = open("track_id_with_lyrics", 'w')

  	# If no number of songs is given, search entire file.
	if len(sys.argv) < 2:
		numSongs = float("inf")

	else:
		# How many songs we want to find (how many track IDs we want to find).
		numSongs = int(sys.argv[1])

	# Read file of potential track_id values.
	with open('tracks_per_year_desc.txt') as f:
		lines = f.readlines()

        # Number of successful track_id values we have found.
	good_track_id = 0

	# Number of unsuccessful track_id values we have found.
	bad_track_id = 0

	for line in lines:

		if good_track_id > (numSongs - 1):
			break

		# Parse out track_id. This is the track ID given by the
		# Million Song Database. String off newline character on the right.
		track_id = line.rstrip();

		# Read lyrics SQLite database.
		#
		# Comes from the musiXmatch dataset, the official collection of lyrics
		# for the Million Song Dataset from Columbia University.
		#
		# See: http://labrosa.ee.columbia.edu/millionsong/musixmatch
		# for Million Song Dataset
		#
		# See: https://github.com/tbertinmahieux/MSongsDB/blob/master/Tasks_Demos/Lyrics/README.txt
		# for description of lyrics SQLite database.
		conn = sqlite3.connect('mxm_dataset.db')

		# Get BOW lyrics for the given track_id.
		# Lists words in descending frequency (count).
		query = "SELECT track_id, word, count from lyrics WHERE track_id = '" + str(track_id) + "' ORDER BY count DESC"

		cursor = conn.execute(query)

		if len(cursor.fetchall()) == 0:
			print track_id, "was bad"
			bad_track_id += 1

		else:
			print track_id, "was good"
			good_track_id += 1
			target.write(track_id)
			target.write("\n")

		#cursor = conn.execute(query)

		#for row in cursor:
		#	print "track_id = ", row[0]
		#	print "    word = ", row[1]
		#	print "   count = ", row[2], "\n"


		conn.close()

	print "\n   Number with lyrics:", good_track_id
	print "Number without lyrics:", bad_track_id
	print "  Percent with lyrics: %.2f" % (100 * float(good_track_id) / (good_track_id + bad_track_id))

	target.flush()
	target.close()

 
if __name__ == '__main__':
	main()
