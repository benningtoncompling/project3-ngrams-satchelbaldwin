#! /usr/bin/env python3

import sys, math, nltk

def open_files(args):
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    infile = open(inpath, 'r')
    outfile = open(outpath, 'w')
    return(infile, outfile)

def close_files(infile, outfile):
    infile.close()
    outfile.close()


#
# Part 1
#


def tag_sentences(lines):
    sentences = []
    for line in lines:
        sentence = line.split()
        sentence.insert(0, "<s>")
        sentence.append("</s>")
        sentence = [word.lower() for word in sentence]
        sentences.append(sentence)
    return sentences

def get_ngrams(sentences, n):
    ngrams = []
    for s in sentences:
        for ngram in nltk.ngrams(s, n):
            ngrams.append(ngram)
    return(ngrams)

def count_ngrams(ngrams):
    ngram_counts = {}
    for ngram in ngrams:
        if ngram not in ngram_counts:
            ngram_counts[ngram] = 1
        else:
            ngram_counts[ngram] = ngram_counts[ngram] + 1
    return ngram_counts

class NGram:
    def __init__(self, count, probability):
        self.count = count
        self.prob = probability
        self.log_prob = math.log(self.prob)

class NGramData:
    def __init__(self, n):
        self.n = n
        self.ngrams = {}
    def add(self, name, count, probability):
        self.ngrams[name] = NGram(count, probability)

def build_ngram_model(lines):
    sentences = tag_sentences(lines)
    print('tagged sentences!')
    ngram_data = []
    for n in [1, 2, 3]:
        ngrams = get_ngrams(sentences, n)
        print(f'got {n}-grams')
        ngram_counts = count_ngrams(ngrams)

        # conditional frequency is not used with unigrams
        cfd = None
        if n == 2:
            cfd = nltk.ConditionalFreqDist(ngrams)
        if n == 3:
            ngrams_w2 = [ ((n[0], n[1]), n[2]) for n in ngrams ]
            cfd = nltk.ConditionalFreqDist(ngrams_w2)

        data = NGramData(n)
        for ngram in ngrams:
            # unigrams
            count = ngram_counts[ngram]
            if n == 1:
                data.add(' '.join(ngram), count, count / len(ngrams))
            # bigrams
            if n == 2:
                probability = cfd[ngram[0]].freq(ngram[1])
                data.add(' '.join(ngram), count, probability)
            # trigrams
            if n == 3:
                probability = cfd[(ngram[0], ngram[1])].freq(ngram[2])
                data.add(' '.join(ngram), count, probability)

        ngram_data.append(data)
    return ngram_data

# avoid string concatenation! it creates a new copy each time
# *much* faster than the last one
def write_model_to_string(model):
    unigrams = model[0]
    bigrams  = model[1]
    trigrams = model[2]

    get_lines = lambda ngrams: (
        '\n'.join(["{} {} {} {}".format(
            d.count, 
            d.prob, 
            d.log_prob,
            n
            ) for (n, d) in ( 
                sorted(
                    ngrams.items(), 
                    key = lambda t: t[1].count, 
                    reverse = True
                    ))]))

    r = """\
\\data\\
ngram 1: types={} tokens={}
ngram 2: types={} tokens={}
ngram 3: types={} tokens={}

\\1-grams:
{}

\\2-grams:
{}

\\3-grams:
{}""".format(
        len(unigrams.ngrams.keys()),
        sum([c.count for c in unigrams.ngrams.values()]),
        len( bigrams.ngrams.keys()),
        sum([c.count for c in  bigrams.ngrams.values()]),
        len(trigrams.ngrams.keys()),
        sum([c.count for c in trigrams.ngrams.values()]),
        get_lines(unigrams.ngrams),
        get_lines( bigrams.ngrams),
        get_lines(trigrams.ngrams)
        )
    return r


#
# Part 2
#
import random

class Model:
    def __init__(self, text):
        self.unigrams = []
        self.bigrams  = []
        self.trigrams = []
        current_n = None
        for line in text:
            if (line.startswith('\data') or
                line.startswith('ngram') or
                line.strip() == ''):
                continue
            n_set = False
            for n in [1, 2, 3]:
                if line.startswith('\{}-grams'.format(n)):
                    current_n = n
                    n_set = True
            if n_set: continue
            split = line.split()
            # ngram, probaility
            if current_n == 1:
                self.unigrams.append((split[3], float(split[1])))
            if current_n == 2: 
                self.bigrams.append(
                        ( (split[3], split[4]) , float(split[1])) )
            if current_n == 3:
                self.trigrams.append(
                        ( (split[3], split[4], split[5]) , float(split[1])) )

    def random_unigram(self):
        threshold = random.random()
        counter = 0.0
        word = None
        for unigram in self.unigrams:
            word = unigram[0]
            counter = counter + unigram[1]
            if counter > threshold:
                return word
    def generate_unigrams(self, n):
        sentences = []
        for _ in range(n):
            words = ['<s>']
            while words[-1] != '</s>':
                word = self.random_unigram()
                while word == '<s>':
                    word = self.random_unigram()
                words.append(word)
            sentences.append(words)
        return sentences

    def random_word_from_bigram(self, starting_word):
        threshold = random.random()
        counter = 0.0
        current_bigram = None
        for bigram in self.bigrams:
            current_bigram = bigram[0]
            if current_bigram[0] == starting_word:
                counter = counter + bigram[1]
                if counter > threshold:
                    return current_bigram[1]
    def generate_bigrams(self, n):
        sentences = []
        for _ in range(n):
            words = ['<s>']
            while words[-1] != '</s>':
                word = self.random_word_from_bigram(words[-1])
                while word == '<s>':
                    word = self.random_word_from_bigram(words[-1])
                words.append(word)
            print(f'{words}')
            sentences.append(words)
        return sentences

    def generate_trigrams():
        pass




def generate_from_ngram(model_text): 
    model = Model(model_text)
    model.generate_unigrams(2)
    model.generate_bigrams(2)
    generated = {}
    return generated

def format_generated(generated):
    pass

#
# Part 3
#


def b(): 
    pass
