import urllib.request
from HTMLParser import HTMLParser
import os
import pickle
import concurrent.futures

def get_html(url):
    """
    Gets the HTML content from URL
    :param url: place to get HTML from
    :return: the HTML content
    """
    print("Fetching URL: ", url)
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read()
    except urllib.error.HTTPError:
        html = "404 Error"
    if type(html) is not str:
        return html.decode("utf-8", "ignore")
    else:
        return html

if __name__ == '__main__':

    parser = HTMLParser()
    parsed_HTML = parser.parse(get_html("http://www.cs.ucl.ac.uk"), bag_of_words=True)

    path = "Evaluation/Queries_and_Google_URLs/"
    for fname in os.listdir(path):

        query_save_path = path + "Pickles/" + fname.strip(".txt") + ".pkl"

        query_dict = {}
        urls = []
        with open(path + fname, "r") as f:
            for url in f.readlines():
                url = url.strip()
                urls.append(url)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(get_html, url): url for url in urls}

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                data = parser.parse(future.result(), bag_of_words=True)
                query_dict[url] = data["bag_of_words"]

        with open(query_save_path, "wb") as f:
            pickle.dump(query_dict, f)

        for key, val in query_dict.items():
            print("KEY: ", key)
            #print("VAL: ", val)

        #input("Press enter to continue...")