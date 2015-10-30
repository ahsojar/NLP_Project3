import operator
import os
import os.path
import nltk
from nltk.corpus import stopwords

#Output file name
kaggleTest = "kaggleTest.csv"

def training(fname):
  with open(fname) as f:
    content = f.readlines()
    lexicon = {}
    for l in range(0,len(content),3):
      words = content[l].split()
      pos = content[l+1].split()
      tags = content[l+2].split()

      for w in range(0,len(words)):
        if words[w] not in lexicon:
          lexicon[words[w]] = {}
          lexicon[words[w]][tags[w]]= 1
        else:
          if tags[w] in lexicon[words[w]]:
            lexicon[words[w]][tags[w]] += 1
          else:
            lexicon[words[w]][tags[w]] = 1

  return lexicon


def test(fname,lexicon):
  predictions = {
  'PER':[],
  'LOC':[],
  'ORG':[],
  'MISC':[]
}
  with open(fname) as f:
    content = f.readlines()
    for l in range(0,len(content),3):
      words = content[l].split()
      pos = content[l+1].split()
      number = content[l+2].split()
      lasttag = 0
      for w in range(0,len(words)):
        word = words[w]
        if word in lexicon:
          tag = lexicon[word]
          #select the tag that the word was most used as in the training corpus
          maxtag = max(tag.iteritems(), key=operator.itemgetter(1))[0]
          if maxtag != 'O':
            shorten = maxtag[2:]
            predictions[shorten].append(number[w])
  print_to_file(predictions)
  return predictions

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
      f.write(start + "-")
      for w in range(1,len(predictions[k])):
        if not (int(predictions[k][w]) - int(start) == 1):
          f.write(start + " ")
          start = predictions[k][w]
          f.write(start + "-")
      f.write(start+"\n")

lexicon = training('train.txt')
test('test.txt', lexicon)