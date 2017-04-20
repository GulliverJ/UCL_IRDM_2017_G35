# Create a link to use UCL search on cs subdomain from a list of queries
# Example: alan turing -> http://search2.ucl.ac.uk/s/search.html?query=alan+turing&Submit=&collection=ucl-public-meta&subsite=cs

links = []

with open('queries.txt') as f_in:

    for line in f_in.readlines():

        link = 'http://search2.ucl.ac.uk/s/search.html?query='
        link += line.strip().replace(' ', '+')
        link += '&Submit=&collection=ucl-public-meta&subsite=cs'

        links.append(link)

for link in links:
    print(link)