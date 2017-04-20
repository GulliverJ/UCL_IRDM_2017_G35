from flask import Flask
from flask import request
from flask import render_template
import hits
from rankers.VSMRanker import VSMRanker
import pickle
import time

app = Flask(__name__)

listing_conversion = None

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

def translate_ranking(ranking):
	"""
	Translate numerical page IDs to readable info for display
	:param ranking: a list of document IDs, with the ID at position 0 corresponding to the most highly ranked file.
	:return: dictionaries of information corresponding to the IDs, used for display
	"""
	listings = []
	for item in ranking:
		listings.append( listing_conversion[ "%d.json" % (item) ] )
	# NOTE: Enforcing only top 20 listings for now?
	return listings[0:20]

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
