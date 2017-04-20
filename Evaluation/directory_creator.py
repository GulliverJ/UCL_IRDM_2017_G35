import os

WRITE_PATH = "EvaluationData/Queries_and_Google_URLs/"

counter = 1
while True:
    prompt = "Enter the file name for file " + str(counter) + ": "
    name = input(prompt)

    name_tokens = name.split()
    final_name = ""
    for i, token in enumerate(name_tokens):
        if i == len(name_tokens) - 1:
            final_name += token
        else:
            final_name += token + "_"

    with open(WRITE_PATH + final_name + ".txt", "w") as f:
        pass

    counter += 1