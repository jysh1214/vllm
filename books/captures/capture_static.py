#!/usr/bin/env python
"""Static ground-truth captures for the vLLM book (no GPU needed)."""
import os

os.environ["PYTHONHASHSEED"] = "0"  # reproducible NONE_HASH

import vllm

print("=== [C1] version ===")
print(f"vllm {vllm.__version__}")

print("\n=== [C2] RequestStatus (IntEnum order is load-bearing) ===")
from vllm.v1.request import RequestStatus

for s in RequestStatus:
    fin = "finished" if RequestStatus.is_finished(s) else ""
    print(f"{int(s):3d}  {s.name:42s} {fin}")

print("\n=== [C3] SchedulerOutput fields ===")
import dataclasses
from vllm.v1.core.sched.output import SchedulerOutput

for f in dataclasses.fields(SchedulerOutput):
    print(f"- {f.name}")

print("\n=== [C4] chained block hash demo (hash_block_tokens) ===")
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash

init_none_hash(sha256_cbor)
blk_A1 = list(range(0, 16))       # request A, block 1 tokens
blk_A2 = list(range(16, 32))      # request A, block 2 tokens
blk_B1 = list(range(100, 116))    # request B, DIFFERENT block 1

hA1 = hash_block_tokens(sha256_cbor, None, blk_A1)
hA2 = hash_block_tokens(sha256_cbor, hA1, blk_A2)
hB1 = hash_block_tokens(sha256_cbor, None, blk_B1)
hB2 = hash_block_tokens(sha256_cbor, hB1, blk_A2)  # same local tokens as A2!

print(f"h(A1)               = {hA1.hex()[:16]}…")
print(f"h(A2 | parent=A1)   = {hA2.hex()[:16]}…")
print(f"h(B1)               = {hB1.hex()[:16]}…")
print(f"h(A2 | parent=B1)   = {hB2.hex()[:16]}…")
print(f"same local tokens, different prefix -> equal? {hA2 == hB2}")

print("\n=== [C5] Inductor CPU VecISA list (RVV present?) ===")
import torch._inductor.cpu_vec_isa as cvi

isas = [type(i).__name__ for i in cvi.supported_vec_isa_list]
print(f"supported_vec_isa_list classes: {isas}")
print(f"any RVV / riscv: {any('rvv' in n.lower() or 'riscv' in n.lower() for n in isas)}")

print("\n=== [C6] torch / platform ===")
import torch

print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
