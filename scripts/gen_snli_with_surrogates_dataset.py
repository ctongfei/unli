from typing import *
import argparse

parser = argparse.ArgumentParser(description="Generate SNLI with surrogate training.")
parser.add_argument("--input", type=str, default="", help="Path to SNLI (QRels format)")
parser.add_argument("--surrogate", type=str, default="", help="File to surrogate scores")
ARGS = parser.parse_args()

surrogate_scores = {}
for l in open(ARGS.surrogate, mode='r'):
    y, s = l.strip().split("\t")
    surrogate_scores[y] = float(s)

for l in open(ARGS.input, mode='r'):
    pid, _, hid, y = l.strip().split("\t")
    u = surrogate_scores.get(y, surrogate_scores["1"])  # default is neutral
    print(f"{pid}\tITER\t{hid}\t{u}")
