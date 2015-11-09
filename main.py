import operator
import os
import os.path
import nltk
from nltk.corpus import stopwords
import numpy as np
from hmm import HMM

#Output file name
kaggleTest = "kaggleTest.csv"


unigram_counts = {} 
bigram_counts = {} #2D bigram counts {word1: {word2: count, word3: count}, word2: {word4: count}}
bigram_prob = {} #probability of each bigram, 2D

def training(fname):
  with open(fname) as f:
    content = f.readlines()
    word_tag_count = {}
    tag_counts= {'<S>': 0,'I-PER': 0,'B-PER':0, 'O':0, 'I-LOC':0,'B-LOC':0, 'I-ORG':0, 'B-ORG':0, 'I-MISC':0,'B-MISC':0}

    for l in range(0,len(content),3):
      words =['<S>'] + content[l].split()
      pos = ['<S>'] + content[l+1].split()
      tags = ['<S>']+ content[l+2].split()
      bigram(tags)
      for w in range(0,len(words)):
        tag_counts[tags[w]] += 1
        if words[w] not in word_tag_count:
          word_tag_count[words[w]] = {'<S>': 0, 'I-PER': 0, 'B-PER':0,'O':0, 'I-LOC':0, 'B-LOC':0, 'I-ORG':0, 'B-ORG':0, 'I-MISC':0,'B-MISC':0}
          word_tag_count[words[w]][tags[w]]= 1
        else:
          word_tag_count[words[w]][tags[w]] += 1
  transition_prob = normalize_bigram(unigram_counts, bigram_counts) 
  emission_prob = normalize_emission(word_tag_count, tag_counts)
  return [transition_prob, emission_prob]

def test(fname,word_tag_count):
  predictions = {
  'PER':[],
  'LOC':[],
  'ORG':[],
  'MISC':[]
}
  with open(fname) as f:
    content = f.readlines()
    for l in range(0,len(content),3):
      pass
      # words = content[l].split()
      # pos = content[l+1].split()
      # number = content[l+2].split()
      # lasttag = 0
      # for w in range(0,len(words)):
      #   word = words[w]
      #   if word in word_tag_count:
      #     tag = word_tag_count[word]
      #     #select the tag that the word was most used as in the training corpus
      #     maxtag = max(tag.iteritems(), key=operator.itemgetter(1))[0]
      #     if maxtag != 'O' or maxtag!='<S>':
      #       shorten = maxtag[2:]
      #       predictions[shorten].append(number[w])
  print predictions
 # print_to_file(predictions)
  return predictions


def bigram(tokens):
  for word in tokens:
    #Create unigram counts
    if word in unigram_counts:
      unigram_counts[word] += 1
    else:
      unigram_counts[word] = 1
  prev_word = '<S>'
  for word in tokens:
    #create 2 dimensional bigrams
    if prev_word in bigram_counts:
      if word in bigram_counts[prev_word]:
        bigram_counts[prev_word][word] += 1
      else:
        bigram_counts[prev_word][word] = 1
    else:
      bigram_counts[prev_word] = {}
      bigram_counts[prev_word][word] = 1

    prev_word = word

def normalize_bigram(unigram_counts, bigram_2d):
  for first_word in bigram_counts:
    second_words = bigram_counts[first_word]
    normalized_words = {}
    for w in second_words:
      if w != '<S>':
        normalized_words[w] = second_words[w]/ float(unigram_counts[first_word])
        bigram_prob[first_word] = normalized_words
  return bigram_prob

def normalize_emission(emission_counts, tag_counts):
  for all_tags in emission_counts:
    for t in emission_counts[all_tags]:
      emission_counts[all_tags][t] = emission_counts[all_tags][t]*(1.0/tag_counts[t])
  return emission_counts

#the Viterbi algorithm
def viterbi(hmm, initial_dist, emissions):
    probs = hmm.emission_dist(emissions[0]) * initial_dist
    stack = []

    for emission in emissions[1:]:
        trans_probs = hmm.transition_probs * np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = hmm.emission_dist(emission) * trans_probs[max_col_ixs, np.arange(hmm.num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq

def print_to_file(predictions):
  if os.path.exists(kaggleTest):
    mode = 'a'  
  else: 
    mode = 'a'
    with open(kaggleTest, mode) as f:
      f.write("Type,Prediction\n")
  with open(kaggleTest, mode) as f:
    for k in predictions:
      f.write(k + ", ")
      start = predictions[k][0]
      f.write(str(start) + "-")
      for w in range(1,len(predictions[k])):
        if not (int(predictions[k][w]) - int(start) == 1):
          f.write(str(start) + " ")
          start = predictions[k][w]
          f.write(str(start) + "-")
        else:
          start = predictions[k][w]

      f.write(str(start)+"\n")
 
hmm = training('train.txt')
transition_probs = hmm[0]
emission_probs = hmm[1]
hmm_transition_probs = np.array([[transition_probs[w][tag] for tag in sorted(transition_probs[w])] for w in sorted(transition_probs)])
hmm_emission_probs = np.array([[emission_probs[w][tag] for tag in sorted(emission_probs[w])] for w in sorted(emission_probs)]) 
print hmm_transition_probs
# hmm_startprobs = np.array([[0.6, 0.4]])
# wiki_hmm = HMM(wiki_transition_probs, wiki_emission_probs)
# print(viterbi(wiki_hmm, wiki_initial_dist, wiki_emissions))
# # test('test.txt', emission_probs)

