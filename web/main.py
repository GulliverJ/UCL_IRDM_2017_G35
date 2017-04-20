from flask import Flask
from flask import request
from flask import render_template
import hits
from rankers.VSMRanker import VSMRanker
import pickle
import time
from math import sqrt

app = Flask(__name__)

listing_conversion = pickle.load( open( "./data/filenames2results.pkl", "rb"))

#### TODO
# - Index and adjacency matrix should be loaded here for global use
# - Other ranking algorithms need to be added in

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/results", methods=['GET'])
def results():

	# Init rankings and results
	ranking = []
	results = {}

	# Save start time
	start = time.time()

	# Apply the appropriate ranking algorithm
	if(request.args['ranking'] == 'pagerank'):
		## TODO get ranking from pagerank search method
		results['num'] = len(ranking)
		results['items'] = translate_ranking(ranking)

	elif(request.args['ranking'] == 'hits'):
		hubs, auths = hits.search(request.args['query'])

		results['num'] = len(hubs) + len(auths)

		# Translate rankings for both hubs and authorities
		results['hubs'] = translate_ranking(hubs)
		results['authorities'] = translate_ranking(auths)

	elif(request.args['ranking'] == 'vsm'):
		# TODO vsm ranking, e.g.
		ranking = vsm_model.search(request.args['query'])
		results['num'] = len(ranking)
		results['items'] = translate_ranking(ranking)

	else:
		# ranking = lsi.search(request.args['query'])
		results['num'] = len(ranking)
		results['items'] = translate_ranking(ranking)

	# Store the time taken to retrieve
	results['time'] = "%.3f" % (time.time() - start)

	# Pass that into web page template to render results
	return render_template("index.html", results=results)


def similarity(bow1, bow2):
    """
    Compares file similarity on two bag of words representations
    :param bow1: first bag of words
    :param bow2: second bag of words
    :return: similarity between 0 and 1
    """
    dot_prod, file1_norm, file2_norm = 0.0, 0.0, 0.0
    for word, count1 in bow1.items():
        count2 = bow2[word]
        dot_prod += count1 * count2
        file1_norm += count1 ** 2
        file2_norm += count2 ** 2
    norm = sqrt(file1_norm * file2_norm)
    if norm > 0:
        return dot_prod / norm
    return 0


def translate_ranking(ranking):
	"""
    Translate numerical page IDs to readable info for display - only keeping pages which are not already present.
    :param ranking: a list of document IDs, with the ID at position 0 corresponding to the most highly ranked file.
    :return: dictionaries of information corresponding to the IDs, used for display
    """
	# The final results to display
	listings = []

	for item in ranking:

		# Get the result for the first PID
		file_result = listing_conversion["%d.json" % item]

		if not file_result:
			continue

		# This will be true if the current item is already in the ranking
		already_present = False

		# Check the existing list of files
		for existing_listing in listings:

			# Calculate similarity using vector methods
			sim = similarity(file_result["bag_of_words"], existing_listing["bag_of_words"])

			# If similarity is more than 90%, it is likely the file is already present
			if sim > 0.9:
				already_present = True
				break

		# Only append a file if it is not already there
		if not already_present:
			listings.append(file_result)

	# NOTE: Enforcing only top 20 listings for now?
	num_results = 20

	if len(listings) > num_results:
		return listings[0:num_results]
	else:
		return listings

if __name__ == "__main__":

    # HITS
    listing_conversion = pickle.load( open( "./data/filenames2results.pkl", "rb"))

    # VSM
    with open("./data/filename2url_vsm.pkl", "rb") as f:
        load = pickle.load(f)
        vsm_filenames = load.keys()
    with open("./data/inverted_index_vsm.pkl", "rb") as f:
        load = pickle.load(f)
        vsm_index = load[0]
    with open("./data/word_counts_vsm.pkl", "rb") as f:
        vsm_word_counts = pickle.load(f)
    variant = "ntc.ltc"
    vsm_model = VSMRanker(vsm_index, vsm_word_counts, vsm_filenames, variant)

    app.run()
