import numpy as np
import pandas as pd
import os
from random import random

LEN_OF_SEQ = 100
SEQUENCE_NUMBERS = range(1, 11)
NUM_SEQS = len(SEQUENCE_NUMBERS)
NUM_SEQ_TO_GENERATE = 100
BASE_PATH = "dataset/sequences/"

# First: do the basic thing --> grab sequences of 100 frames
# Then, do the more complex thing (100 meters) --> see KITTI evaluation code 

with open("sequences.csv", "a") as sequences:
	for i in range(NUM_SEQ_TO_GENERATE):
		# Randomly pick an index between 0, len(SEQUENCE_NUMBERS)
		seq = int(random() * NUM_SEQS)

		# "Navingate" into the directory, check the number of lines in times.txt
		num_lines = sum(1 for line in open(os.path.join(BASE_PATH, "{:02d}".format(SEQUENCE_NUMBERS[seq]), "times.txt")))

		# Sample a point between 0 and 100 before the last time in times.txt
		start_frame = int(random() * (num_lines - LEN_OF_SEQ)) - 1

		# Randomly pick image_2, or image_3
		camera = int(random() * 2) + 2

		# Append line to file: "{:2d},{},{:5d},{}".format(seq,camera,start_frame,LEN_OF_SEQ)
		sequences.write("{:02d},image_{:1d},{:06d},{}\n".format(SEQUENCE_NUMBERS[seq],camera,start_frame,LEN_OF_SEQ))

