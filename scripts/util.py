from typing import *


def read_dict(f: str) -> Dict[str, str]:
    d = {}
    for l in open(f, 'r'):
        sid, sentence = l.strip().split("\t")
        d[canonicalize(sentence)] = sid
    return d


def read_qrels(f: str) -> Dict[Tuple[str, str], str]:
    d = {}
    for l in open(f, 'r'):
        pid, _, hid, cl = l.strip().split("\t")
        d[(pid, hid)] = cl
    return d


def canonicalize(s: str):
    return "".join(filter(lambda c: c.isalnum(), s))
