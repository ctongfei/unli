import csv
import argparse
import sys
from typing import *
from util import *
import os

os.chdir(sys.path[0])

parser = argparse.ArgumentParser(description="UNLI to QRels format")
parser.add_argument("--snli", type=str, default="", help="Path to SNLI (QRels format)")
parser.add_argument("--usnli_train", type=str, default="", help="Path to UNLI train (CSV format)")
parser.add_argument("--usnli_dev", type=str, default="", help="Path to UNLI dev (CSV format)")
parser.add_argument("--usnli_test", type=str, default="", help="Path to UNLI test (CSV format)")
parser.add_argument("--out", type=str, default="", help="Output directory")
ARGS = parser.parse_args()


def process(lines, partition: str):
    pre_dict = read_dict(f"{ARGS.snli}/{partition}.l")
    hyp_dict = read_dict(f"{ARGS.snli}/{partition}.r")

    table = {}

    for row in lines:
        _, pre, hyp, cl, u = row
        pre_canon = canonicalize(pre)
        hyp_canon = canonicalize(hyp)
        pid = pre_dict[pre_canon]
        hid = hyp_dict[hyp_canon]

        if pid not in table:
            table[pid] = []

        table[pid].append((hid, u))

    qrels = open(f"{ARGS.out}/{partition}.qrels", mode='w')
    for pid, hys in table.items():
        for hid, y in hys:
            print(f"{pid}\tITER\t{hid}\t{y}", file=qrels)
    qrels.close()


train = list(csv.reader(open(ARGS.usnli_train, 'r')))[1:]
dev = list(csv.reader(open(ARGS.usnli_dev, 'r')))[1:]
test = list(csv.reader(open(ARGS.usnli_test, 'r')))[1:]

process(train, "train")
process(dev, "dev")
process(test, "test")
