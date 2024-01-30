'''
Project Title:  CollegeBot An Intelligent System
Created By:     CollegeBot Team @ K K Wagh Polytechnic
Leader:         Aditya Chaphekar
Last Edited:    22/02/2020
'''


import warnings
#using the warnings library for removing the Deprecrated Module Warnings
with warnings.catch_warnings():
    # importing required Modules
    warnings.filterwarnings("ignore")
    import tensorflow
    tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
    import tflearn
    import nltk
    from nltk.stem.lancaster import LancasterStemmer
    import numpy
    import random
    import json
    from flask import Flask, render_template, request
    import pickle
    import os




# initializing the FLASK and stemmer Objects.
app = Flask(__name__)
stemmer = LancasterStemmer()

#loading the intent file {database}
with open("intents.json", encoding="utf8") as file:
    data = json.load(file)

try:
    #open the data.pickle if available and initialize variables
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    #if the file is not available initialise empty list
    words = []
    labels = []
    docs_x = []
    docs_y = []

    #assign values in the list fetched from the json file
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    #creating an list/bag of stemmed words
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()
#initializing training variables
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#trainig the bot
model.fit(training, output, n_epoch=2000, batch_size=10, show_metric=True)
model.save("model.tflearn")

#function to create the basic form of each word
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

# the main function
def chat(inp1):
    inp = inp1
    #exit the chatbot if the input is quit
    if inp.lower() == "quit":
        return "thanks for visiting"
    #search for the question, if not available search for related question
    results = model.predict([bag_of_words(inp, words)])
    #choose the keyword with highest weight
    results_index = numpy.argmax(results)
    #choosing the tag
    tag = labels[results_index]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
    #return random response
    return(random.choice(responses))


@app.route("/")
def home():
    #displaying the index page
    return render_template("index.html")


@app.route("/train")
def train():
    #displaying the training page & training the bot
    os.remove("checkpoint")
    os.remove("data.pickle")
    os.remove("model.tflearn.data-00000-of-00001")
    os.remove("model.tflearn.index")
    os.remove("model.tflearn.meta")
    model.fit(training, output, n_epoch=2000, batch_size=10, show_metric=True)
    model.save("model.tflearn")
    return render_template("training_Successful.html")


@app.route("/get")
def get_bot_response():
    #get response from the user
    return chat(request.args.get('msg'))


if __name__ == "__main__":
    app.run()
