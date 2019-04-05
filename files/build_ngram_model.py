#! /usr/bin/env python3

import sys, nltk
from utilities import *

(infile, outfile) = open_files(sys.argv)

lines = infile.readlines()
model = build_ngram_model(lines)
s = write_model_to_string(model)
outfile.write(s)

close_files(infile, outfile)
