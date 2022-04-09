# library for server
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

length = 0  # length of the datframe
pos_count = 0  # positive_sentiment count
neg_count = 0  # negative_sentiment count

def sentence_to_words(sentence):

    l = sentence.lower()  # convert sentence to lowercase
    l = l.split(' ')  # split sentence into individual word
    p = ''
    word_list = []

    for word in l:

        p = ''

        for letter in word:

            if ord(letter) >= 67 and ord(letter) <= 122:
                p = p + letter
        word_list.append(p)

    return word_list  # return the word list of the sentence devoid of special characters and numericals


def naive_bayes_train(X, Y, a=0.000001):
    n_length = len(X)
    n_class_pos = len(Y[Y == 1])
    n_class_neg = len(Y[Y == 0])
    prior_pos = n_class_pos / n_length  # prior probability for  class
    prior_neg = n_class_neg / n_length  #prior probability for class 
    (n, p, bag) = bag_of_words_maker(X, Y)

    pr = {}

    for i in range(len(bag)):   #evaluating the likelihood prob for each word given a class
        p_pos = (bag['count_pos'][i] + a) / (p + len(bag) * a)

        p_neg = (bag['count_neg'][i] + a) / (n + len(bag) * a)

        pr[bag['index'][i]] = [p_pos, p_neg]
    pr = pd.DataFrame(pr).T
    pr.columns = ['sent=positive', 'sent=negative']
    pr = pr.reset_index()

    return (prior_pos, prior_neg, pr)


def naive_bayes_predict(
    X,
    pr,
    prior_pos,
    prior_neg,
    ):
    Y = []

    for i in range(len(X)):
        k_pos = 1
        k_neg = 1
        p = sentence_to_words(X[i])

        for k in range(len(pr)):

            for word in p:

                if word == pr['index'][k]:
                    k_pos = k_pos * pr['sent=positive'][k] #pdt of likelihood prob given the word is present in vocabulary 
                    k_neg = k_neg * pr['sent=negative'][k]

        nb = [prior_neg * k_neg, prior_pos * k_pos] # multiply each likelihood prob with the prior prob
        Y.append(np.argmax(nb))

    return Y

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
stop_factory = StopWordRemoverFactory()

more_stopword = ["dengan", "ia", "bahwa", "oleh", "[mak"]

custom_stopword = stop_factory.get_stop_words() + more_stopword

stopword = stop_factory.create_stop_word_remover()

def bag_of_words_maker(X, Y):

    bag_dict_binary_NB_pos = {} #keeping track of the positive class words
    bag_dict_binary_NB_neg = {} #keeping track of the negative class words

    stop_words = custom_stopword
    
    for i in range(len(X)):
        p = sentence_to_words(X[i])
        sent = Y[i]
        x_pos = {}
        x_neg = {} #we intialize the dict every iteration so that it does not consider repititions .(Binary NB)

        # print(p)

        if sent == 1:
            for word in p:

                if word in x_pos.keys():
                    x_pos[word] = [x_pos[word][0] + 1, x_pos[word][1]]  #word is the key and value stored is [count, sentiment]
                else:
                    x_pos[word] = [1, sent]

            for key in x_pos.keys():

                if key in bag_dict_binary_NB_pos.keys():
                    bag_dict_binary_NB_pos[key] = \
                        [bag_dict_binary_NB_pos[key][0] + 1,
                         bag_dict_binary_NB_pos[key][1]]
                else:

                    bag_dict_binary_NB_pos[key] = [1, sent]  #storing it in the final dict 

        if sent == 0:

            for word in p:
                if word in x_neg.keys():
                    x_neg[word] = [x_neg[word][0] + 1, x_neg[word][1]]
                else:
                    x_neg[word] = [1, sent]
            for key in x_neg.keys():
                if key in bag_dict_binary_NB_neg.keys():
                    bag_dict_binary_NB_neg[key] = \
                        [bag_dict_binary_NB_neg[key][0] + 1,
                         bag_dict_binary_NB_neg[key][1]]
                else:

                    bag_dict_binary_NB_neg[key] = [1, sent]

    # print(bag_dict_multi.keys())
    # returns the dataframe containg word count in each sentiment 
    neg_bag = pd.DataFrame(bag_dict_binary_NB_neg).T
    pos_bag = pd.DataFrame(bag_dict_binary_NB_pos).T

    neg_bag.columns = ['count_neg', 'sentiment_neg']
    pos_bag.columns = ['count_pos', 'sentiment_pos']
    
    try:
      neg_bag = neg_bag.drop(stop_words)
      pos_bag = pos_bag.drop(stop_words)
    except:
      print('None')
    
    neg_bag = neg_bag.reset_index()
    pos_bag = pos_bag.reset_index()
    n = len(neg_bag)
    p = len(pos_bag)
    bag_of_words = pd.merge(neg_bag, pos_bag, on=['index'], how='outer')
    bag_of_words['count_neg'] = bag_of_words['count_neg'].fillna(0)
    bag_of_words['count_pos'] = bag_of_words['count_pos'].fillna(0)
    bag_of_words['sentiment_neg'] = bag_of_words['sentiment_neg'
            ].fillna(0)
    bag_of_words['sentiment_pos'] = bag_of_words['sentiment_pos'
            ].fillna(1)

    return (n, p, bag_of_words)

def naive_bayes(filename):
	df = pd.read_csv('dataset/' + filename)

	df['Label'].replace(to_replace='Negative', value=0, inplace=True)
	df['Label'].replace(to_replace='Positive', value=1, inplace=True)

	length = len(df)  # length of the datframe
	pos_count = len(df[df['Label'] == 1])  # positive_sentiment count
	neg_count = len(df[df['Label'] == 0])  # negative_sentiment count

	x = df['msg_string']
	y = df['Label']
	(n, p, bag_of_words) = bag_of_words_maker(x, y)

	prior_pos,prior_neg,table = naive_bayes_train(x,y)

	print('prior pos', prior_pos)
	print('prior neg', prior_neg)

	X = df["msg_string"]
	Y = df["Label"]
	x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=.30)

	x_train = x_train.reset_index(drop=True)
	y_train = y_train.reset_index(drop=True)
	y_test = y_test.reset_index(drop=True)
	x_test = x_test.reset_index(drop=True)
	a,b,bag = naive_bayes_train(x_train,y_train)
	y_predicted = naive_bayes_predict(x_test,bag,a,b)

	total_pos = 0
	total_neg = 0

	for y_pred in y_predicted:
		if y_pred == 1:
			total_pos += 1
		else:
			total_neg += 1

	return (total_pos, total_neg)

@cross_origin()
@app.route('/predict', methods=['GET'])
def predict():
	(sinovac_total_pos, sinovac_total_neg) = naive_bayes('vaksin_sinovac_df.csv')
	(astra_total_pos, astra_total_neg) = naive_bayes('vaksin_astra_df.csv')
	# (pfizer_total_pos, pfizer_total_neg) = naive_bayes('vaksin_pfizer_df.csv')
	# moderna = naive_bayes('vaksin_moderna_df.csv')

	sinovac_data = {'positive': sinovac_total_pos, 'negative': sinovac_total_neg}
	astra_data = {'positive': astra_total_pos, 'negative': astra_total_neg}

	temp = {'sinovac': sinovac_data, 'astra': astra_data}

	data = {"data": temp, "status": "ok", "code": 200}

	return jsonify(data), 200

@cross_origin()
@app.route('/about', methods=['GET'])
def about():

  message = "Aplikasi ini berjalan pada flask dengan program versi alpha"
  status = "OK"
  code = 200

  data = {"message": message, "status": status, "code": code}

  return jsonify(data), code

if __name__ == "__main__":
	app.run()