import os
from urllib.parse import parse_qs

for file in os.listdir("raw/"):

    if file.endswith(".txt"):

        urls = []

        with open("raw/" + file, 'r') as f_in:

            for line in f_in.readlines():

                parsed_line = parse_qs(line)
                if parsed_line:
                    urls.append(parsed_line['url'][0])
                else:
                    urls.append(line.strip())

        with open("processed/" + file, 'w') as f_out:

            for url in urls:

                f_out.write("%s\n" % url)
