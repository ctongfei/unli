import csv
import argparse
import sys
from typing import *
from util import *

parser = argparse.ArgumentParser(description="SNLI to QRels format")
parser.add_argument("--snli", type=str, default="", help="Original SNLI format")
parser.add_argument("--out", type=str, default="", help="Output QRels path")
ARGS = parser.parse_args()


def preprocess_partition(partition: str):

    prefix = f"SNLI-{partition}"
    pre_to_pid = {}
    hyp_to_hid = {}

    with \
        open(f"{ARGS.out}/{partition}.l", mode='w') as pre_out, \
        open(f"{ARGS.out}/{partition}.r", mode='w') as hyp_out, \
        open(f"{ARGS.out}/{partition}.qrels", mode='w') as qrels_out:

        snli_it = iter(open(f"{ARGS.snli}_{partition}.txt", 'r'))
        next(snli_it)  # skip CSV header row
        for row in snli_it:
            gold, _, _, _, _, pre, hyp, _, _, *_ = row.strip().split("\t")
            y = {
                "entailment": 2,
                "neutral": 1,
                "contradiction": 0,
                "-": "?"
            }[gold]

            pre_canon = canonicalize(pre)
            hyp_canon = canonicalize(hyp)

            if pre_canon in pre_to_pid:
                pid = pre_to_pid[pre_canon]
            else:
                pid = f"{prefix}-P{len(pre_to_pid)}"
                pre_to_pid[pre_canon] = pid
                print(f"{pid}\t{pre}", file=pre_out)

            if hyp_canon in hyp_to_hid:
                hid = hyp_to_hid[hyp_canon]
            else:
                hid = f"{prefix}-H{len(hyp_to_hid)}"
                hyp_to_hid[hyp_canon] = hid
                print(f"{hid}\t{hyp}", file=hyp_out)

            print(f"{pid}\tITER\t{hid}\t{y}", file=qrels_out)


for partition in ["train", "dev", "test"]:
    preprocess_partition(partition)
