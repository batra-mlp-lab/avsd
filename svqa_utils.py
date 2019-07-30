import sys
import numpy as np

def sentence_to_vec(sentence, mode, nlp):
        #assuming the input has already been preprocessed
	if mode == 0:
		vec = np.zeros((1200,))
		tokens = nlp(sentence)
		for i in range(min(3, len(tokens))):
			vec[i*300:(i+1)*300] = tokens[i].vector
		if len(tokens) >= 4:
			for i in range(len(tokens)-3):
				vec[900:] += tokens[i+3].vector
	elif mode == 1:
		vec = np.zeros((300,))
		tokens = nlp(sentence)
		for token in tokens:
			vec += token.vector
	else:
		sys.exit('Incorrect feature Mode')
	return vec	


def multi_sentence_to_vec(input_sentences, mode, nlp):
	if mode == 0:
		vectors = np.zeros(((len(input_sentences),1200)))
	elif mode == 1:
		vectors = np.zeros(((len(input_sentences),300)))
	else:
		sys.exit('Incorrect feature Mode')
	for i in range(len(input_sentences)):
		vectors[i,:] = sentence_to_vec(input_sentences[i], mode, nlp)
	return vectors

def get_most_popular(_list, top_k):
    unq_items = dict()
    for item in _list:
        if item in unq_items:
            unq_items[item] += 1
        else:
            unq_items[item] = 1
    items_freq = []
    for key, value in unq_items.items():
    	temp = (key,value)
    	items_freq.append(temp)
    sorted_by_second = sorted(items_freq, reverse=True, key=lambda tup: tup[1])
    items_unique, freqs = zip(*sorted_by_second)
    return list(items_unique[:top_k]), list(items_unique[top_k:])


def get_data_lists(data):  
    answers = []
    questions = []
    images = []
    for x in range(len(data)):
        images.append(data[x]['image_id'])
        for j in range(10):
            questions.append(data[x]['dialog'][j]['question'])
            answers.append(data[x]['dialog'][j]['answer'])
    return answers, questions, images

