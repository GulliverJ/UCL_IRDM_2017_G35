# Usage Guide

There are two pieces of code here:
* `HTML2Index` takes a directory full of HTML files and turns it into an inverted index. That is, it creates a dictionary of the form:
```
{ word: [ (doc1, count_of_word_in_doc), (doc2, count2), ... ] }
```
 where there is a single entry for every word in the vocabulary of all HTML files in the given directory.
* `HTMLParser` is an object that will take the name of an HTML file that is stored locally and will read the HTML file. It then parses it into a dictionary (items are detailed below). It also has a method which will parse a query into its constituent tokens.

This was developed in a virtual environment. Dependencies are given in "requirements.txt". Install by running `pip3 install -r requirements.txt`.

## HTML2Index
An object can be created by:
```
>>> index_creator = HTML2Index(directory_to_parse)
>>> index = index_creator.get_index()
```
When the object is instantiated, it will create the index. Once the index is created, it can be accessed like any normal dictionary. The method `save_index()` will save the index dictionary as a Pickle file. The created index can then be used like any normal dictionary:
```
>>> index["CS"]
[("research.html", 3), ("index.html", 1)]
```

## HTMLParser
An object can be created by:
```
>>> parser = HTMLParser()
```
This will then be able to parse any HTML files that are stored locally. To do so, pass the name of the HTML file or relative path to it from the current working directory to the `parse()` method as so:
```
>>> file_info = parser.parse("HTML/index.html")
```
You can also pass an optional boolean to the method to create a bag of words representation of the HTML file in addition to what is already returned by default:
```
>>> file_info = parser.parse("HTML/index.html", bag_of_words=True)
```
Finally you can also parse a query into its constituent tokens:
```
>>> query_tokens = parser.parse_query("computer science")
["computer", "science"]
```
The information returned by default from the parse method is:
* `filename`: the name of the HTML file as it is stored locally
* `title`: the title of the HTML page
* `links`: all of the links present on the HTML page with their surrounding text.
* `urls`: all of the raw urls extracted from the links in the HTML page.
* `file_text`: all of the text as a single string from the HTML file