import sqlite3
import sys

def main():


	# Parse out track_id from command line. This is the track ID given by the
	# Million Song Database.
	track_id = sys.argv[1];
	
	print "Retrieving BOW lyrics for track_id =", track_id, "\n"

	# Read lyrics SQLite database.
	#
	# This is from the musiXmatch dataset, the official collection of lyrics
	# for the Million Song Dataset from Columbia University.
	#
	# See: http://labrosa.ee.columbia.edu/millionsong/musixmatch
	# for Million Song Dataset
	#
	# See: https://github.com/tbertinmahieux/MSongsDB/blob/master/Tasks_Demos/Lyrics/README.txt
	# for description of lyrics SQLite database.
	conn = sqlite3.connect('mxm_dataset.db')

	# Get BOW lyrics for the given track ID.
	# Lists words in descending frequency (count).
	query = "SELECT track_id, word, count from lyrics WHERE track_id = '" + str(track_id) + "' ORDER BY count DESC"
	cursor = conn.execute(query)

	for row in cursor:
		print "track_id = ", row[0]
		print "    word = ", row[1]
		print "   count = ", row[2], "\n"


	conn.close()

 
if __name__ == '__main__':
	main()