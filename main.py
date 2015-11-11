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
bigram_prob = {}
start_tag_counts = {'<S>': 1,'I-PER': 0,'B-PER':0, 'O':0, 'I-LOC':0,'B-LOC':0, 'I-ORG':0, 'B-ORG':0, 'I-MISC':0,'B-MISC':0}

def train(fname):
  with open(fname) as f:
    content = f.readlines()
    num_sentences = len(content)/3.0
    word_tag_count = {'<UNK>':{'<S>': .01,'I-PER': .01,'B-PER':.01, 'O':.01, 'I-LOC':.01,'B-LOC':.01, 'I-ORG':.01, 'B-ORG':.01, 'I-MISC':.01,'B-MISC':.01}}
    tag_counts= {'<S>': .01,'I-PER': .01,'B-PER':.01, 'O':.01, 'I-LOC':.01,'B-LOC':.01, 'I-ORG':.01, 'B-ORG':.01, 'I-MISC':.01,'B-MISC':.01}

    for l in range(0,len(content),3):
      words =['<S>'] + content[l].split()
      pos = ['<S>'] + content[l+1].split()
      tags = ['<S>']+ content[l+2].split()
      #see which tag this sentence starts with, update count
      start_tag_counts[tags[1]] += 1
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
  start_prob = normalize_startprob(start_tag_counts, num_sentences)
  return [transition_prob, emission_prob, start_prob]

def test(fname,start_probs,transition_probs, emission_probs):
  predictions = {
  'PER':[],
  'LOC':[],
  'ORG':[],
  'MISC':[]
}
  with open(fname) as f:
    content = f.readlines()
    for l in range(0,len(content),3):
      #print "working on example ", l/3.0
      words = ['<S>'] + content[l].split()
      number = ['<S>'] + content[l+2].split()
      ner_tags = viterbi(words, ['<S>','I-PER','B-PER', 'O', 'I-LOC','B-LOC', 'I-ORG', 'B-ORG', 'I-MISC','B-MISC'], start_probs, transition_probs, emission_probs)[1]
      for w in range(0,len(ner_tags)):
          tag = ner_tags[w]
          if (tag != 'O' and tag != '<S>' and number[w]!='<S>'):
            shorten = tag[2:]
            predictions[shorten].append(number[w])
  print "predictions"
  print predictions
  print_to_file(predictions)

  return predictions

def bigram(tokens):
  for word in tokens:
    if word in unigram_counts:
      unigram_counts[word] += 1
    else:
      unigram_counts[word] = 1
  prev_word = '<S>'

  for word in tokens:
    if prev_word in bigram_counts:
        bigram_counts[prev_word][word] += 1
    else:
      bigram_counts[prev_word] = {'<S>': 0,'I-PER': 0,'B-PER':0, 'O':0, 'I-LOC':0,'B-LOC':0, 'I-ORG':0, 'B-ORG':0, 'I-MISC':0,'B-MISC':0}
      bigram_counts[prev_word][word] = 1

    prev_word = word

def normalize_startprob(start_tag_counts, num_sentences):
  for w in start_tag_counts:
      start_tag_counts[w] = start_tag_counts[w]*(1.0/num_sentences)
  return start_tag_counts

def normalize_bigram(unigram_counts, bigram_2d):
  for first_word in bigram_counts:
    second_words = bigram_counts[first_word]
    normalized_words = {}
    for w in second_words:
      normalized_words[w] = second_words[w]/ float(unigram_counts[first_word])
      bigram_prob[first_word] = normalized_words
  return bigram_prob

def normalize_emission(emission_counts, tag_counts):
  for all_tags in emission_counts:
    for t in emission_counts[all_tags]:
      emission_counts[all_tags][t] = emission_counts[all_tags][t]*(1.0/tag_counts[t])
  return emission_counts


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
 
def viterbi(obs, states, start_p, trans_p, emit_p):
  # Initialize base cases (t == 0)
  V = [{y:(start_p[y] * emit_p[obs[0]][y]) for y in states}]
  path = {y:[y] for y in states}
  for y in states:
    if obs[0] in emit_p:
      V[0][y] = start_p[y] * emit_p[obs[0]][y]
    else:
      V[0][y] = start_p[y] * emit_p['<UNK>'][y]
    path[y] = [y]
   # Run Viterbi for t > 0
  for t in range(1, len(obs)):
    V.append({})
    newpath = {}
    for y in states:
      if obs[t] in emit_p:
        (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[obs[t]][y], y0) for y0 in states)
      else: 
        (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p['<UNK>'][y], y0) for y0 in states)
      V[t][y] = prob
      newpath[y] = path[state] + [y]
       # Don't need to remember the old paths
    path = newpath
  (prob, state) = max((V[t][y], y) for y in states)
  return (prob, path[state])

hmm = train('train.txt')
transition_probs = hmm[0]
emission_probs = hmm[1]
start_probs = hmm[2]

test('test.txt', start_probs,transition_probs,emission_probs)


