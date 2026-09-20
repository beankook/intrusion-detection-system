"""Generate a PLACEHOLDER dataset with the UNSW-NB15 column layout.

This is random data with class-dependent shifts. It exists only so the pipeline can be
smoke-tested without downloading the real dataset. Results obtained on it say nothing about
intrusion detection - never report them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss",
           "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack",
           "ackdat", "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
           "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
           "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"]
# same proportions as the real training partition, scaled down
CLASSES = {"Normal": 56000, "Generic": 40000, "Exploits": 33393, "Fuzzers": 18184, "DoS": 12264,
           "Reconnaissance": 10491, "Analysis": 2000, "Backdoor": 1746, "Shellcode": 1133, "Worms": 130}


def make(scale: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centers = {c: rng.normal(0, 1.0, len(NUMERIC)) for c in CLASSES}
    frames = []
    for c, n_full in CLASSES.items():
        n = max(40, int(n_full * scale))
        num = np.exp(centers[c] + rng.normal(0, 1.2, (n, len(NUMERIC))))   # heavy-tailed, overlapping
        df = pd.DataFrame(num, columns=NUMERIC)
        df["proto"] = rng.choice(["tcp", "udp", "arp", "ospf"], n, p=[.6, .3, .05, .05])
        df["service"] = rng.choice(["-", "http", "dns", "ftp", "smtp"], n)
        df["state"] = rng.choice(["FIN", "INT", "CON", "REQ"], n)
        df["attack_cat"] = c
        df["label"] = int(c != "Normal")
        frames.append(df)
    out = pd.concat(frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    out.insert(0, "id", np.arange(1, len(out) + 1))
    return out[["id", "dur", "proto", "service", "state"] + NUMERIC[1:] + ["attack_cat", "label"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/DEMO_placeholder.csv")
    ap.add_argument("--scale", type=float, default=0.05)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make(a.scale).to_csv(a.out, index=False)
    print(f"wrote {a.out}")
