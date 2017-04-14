from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
@app.route("/")
def hello():
	return render_template("index.html")

@app.route("/results", methods=['GET', 'POST'])
def results():
    results = searchForResults(request.args['query'])
    return render_template("index.html", results=results)


def searchForResults(query):
    res = [
        { "title": "Result 1", "href": "http://www.cs.ucl.ac.uk", "blurb": "CS main page"},
        { "title": "Result 2", "href": "http://www.cs.ucl.ac.uk/prospective_students", "blurb": "Prospective Students"},
        { "title": "Result 3", "href": "http://www.cs.ucl.ac.uk/careers", "blurb": "Careers page"},
    ]
    return res


if __name__ == "__main__":
	app.run()
