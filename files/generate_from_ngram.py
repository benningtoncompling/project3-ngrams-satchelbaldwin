#! /usr/bin/env python3

import sys, nltk
from utilities import *

(infile, outfile) = open_files(sys.argv)

model = infile.readlines()
generated = generate_from_ngram(model)
formatted = format_generated(generated)
outfile.write(formatted)

close_files(infile, outfile)
