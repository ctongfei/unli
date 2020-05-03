from typing import *
from util import *
import argparse
import statistics

parser = argparse.ArgumentParser(description="Compute UNLI median for the 3 classes (ent/neu/con)")
parser.add_argument("--snli", type=str, default="", help="Path to SNLI (QRels format)")
parser.add_argument("--usnli", type=str, default="", help="Path to u-SNLI train (QRels format)")
parser.add_argument("--aggregator", type=str, default="median", help="Method to aggregate the scores {median/mean}")
ARGS = parser.parse_args()

snli_cl = {
    **read_qrels(f"{ARGS.snli}/train.qrels"),
    **read_qrels(f"{ARGS.snli}/dev.qrels"),
    **read_qrels(f"{ARGS.snli}/test.qrels")
}

cl_scores = {
    "0": [],
    "1": [],
    "2": []
}

for l in open(f"{ARGS.usnli}", mode='r'):
    pid, _, hid, u = l.strip().split("\t")
    y = snli_cl[(pid, hid)]
    if y in cl_scores:
        cl_scores[y].append(float(u))

aggregator = {
    "median": statistics.median,
    "mean": statistics.mean
}[ARGS.aggregator]

for cl, scores in cl_scores.items():
    try:
        agg = aggregator(scores)
    except statistics.StatisticsError:
        agg = {
            "0": 0.0,
            "2": 1.0
        }[cl]
    if cl == "0" and agg > 0.2:
        agg = 0.0
    elif cl == "2" and agg < 0.8:
        agg = 1.0
    print(f"{cl}\t{agg}")
