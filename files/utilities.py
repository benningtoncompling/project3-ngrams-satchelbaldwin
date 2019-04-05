#! /usr/bin/env python3

import sys, math

def open_files(args):
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    infile = open(inpath, 'r')
    outfile = open(outpath, 'w')
    return(infile, outfile)

def close_files(infile, outfile):
    infile.close()
    outfile.close()

def build_ngram_model(lines):
    import nltk
    sentences = []
    for line in lines:
        sentence = line.split()
        sentence.insert(0, "<s>")
        sentence.append("</s>")
        sentence = [word.lower() for word in sentence]
        sentences.append(sentence)
    
    ngram_data = []
    for n in [1, 2, 3]:
        data = {}
        data["n"] = n

        ngrams = []
        for s in sentences:
            for ngram in nltk.ngrams(s, n):
                ngrams.append(ngram)
        
        types = []
        ngram_counts = {}
        for ngram in ngrams:
            if ngram not in types:
                types.append(ngram)
                ngram_counts[ngram] = 1
            else:
                ngram_counts[ngram] = ngram_counts[ngram] + 1
        
        data["ngrams"] = {}
        data["types"] = len(types)
        data["tokens"] = len(ngrams)
        
        cfd = 0
        if n == 2:
            cfd = nltk.ConditionalFreqDist(ngrams)
        if n == 3:
            ngrams_w2 = [ ((n[0], n[1]), n[2]) for n in ngrams ]
            cfd = nltk.ConditionalFreqDist(ngrams_w2)
        for ngram in ngrams:
            # unigrams
            if n == 1:
                data["ngrams"][ngram] = {}
                data["ngrams"][ngram]["count"] = ngram_counts[ngram]
                data["ngrams"][ngram]["probability"] = ngram_counts[ngram] / len(ngrams)
                data["ngrams"][ngram]["log_probability"] = math.log(data["ngrams"][ngram]["probability"])
            # bigrams
            if n == 2:
                data["ngrams"][ngram] = {}
                data["ngrams"][ngram]["count"] = ngram_counts[ngram]
                data["ngrams"][ngram]["probability"] = cfd[ngram[0]].freq(ngram[1])
                data["ngrams"][ngram]["log_probability"] = math.log(data["ngrams"][ngram]["probability"])
            # trigrams
            if n == 3:
                data["ngrams"][ngram] = {}
                data["ngrams"][ngram]["count"] = ngram_counts[ngram]
                data["ngrams"][ngram]["probability"] = cfd[(ngram[0], ngram[1])].freq(ngram[2])
                data["ngrams"][ngram]["log_probability"] = math.log(data["ngrams"][ngram]["probability"])
        ngram_data.append(data)
    return ngram_data
        
def write_model_to_string(model):
    r = ""
    r = r + ("\\data\\") + "\n"
    for item in model:
        r = r + ("ngram {}: type={} token={}".format(item["n"], item["types"], item["tokens"])) + "\n"
    r = r + ("\n")
    for item in model:
        r = r + ("\\{}-grams:".format(item["n"])) + "\n"
        l = sorted(item["ngrams"].items(), key=lambda z: z[1]["count"], reverse = True)
        for n, i in l:
            acc = ""
            for x in n:
                acc = acc + x + " "
            r = r + ("{} {} {} {}".format(i["count"], i["probability"], i["log_probability"], acc)) + "\n"
        r = r + ("\n")
    return (r)
