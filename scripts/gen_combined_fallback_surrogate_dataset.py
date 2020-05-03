from typing import *
import argparse

parser = argparse.ArgumentParser(description="Generate SNLI with surrogate training.")
parser.add_argument("--snli", type=str, default="", help="Path to SNLI QRels file")
parser.add_argument("--usnli", type=str, default="", help="Path to U-SNLI QRels file")
parser.add_argument("--surrogate", type=str, default="", help="File to surrogate scores")
ARGS = parser.parse_args()

surrogate_scores = {}
for l in open(ARGS.surrogate, mode='r'):
    y, s = l.strip().split("\t")
    surrogate_scores[y] = float(s)

unli_items = {}
for l in open(ARGS.usnli, mode='r'):
    pid, _, hid, u = l.strip().split("\t")
    unli_items[(pid, hid)] = u

for l in open(ARGS.snli, mode='r'):
    pid, _, hid, y = l.strip().split("\t")
    u = unli_items.get((pid, hid), surrogate_scores.get(y, surrogate_scores["1"]))  # default is neutral
    print(f"{pid}\tITER\t{hid}\t{u}")
