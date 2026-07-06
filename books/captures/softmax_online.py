#!/usr/bin/env python
"""Online-softmax numeric example for Chapter 4 (matches book's stepper values)."""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
s = np.array([2.0, 1.0, 3.0, 0.5])
V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]])

p = np.exp(s - s.max()); p /= p.sum()
ref = p @ V
print("reference softmax(s)@V =", ref)

m, l, O = -np.inf, 0.0, np.zeros(2)
for lo, hi in [(0, 2), (2, 4)]:
    sb, Vb = s[lo:hi], V[lo:hi]
    m_new = max(m, sb.max())
    alpha = np.exp(m - m_new) if np.isfinite(m) else 0.0
    Pb = np.exp(sb - m_new)
    l = alpha * l + Pb.sum()
    O = alpha * O + Pb @ Vb
    m = m_new
    print(f"after block [{lo}:{hi}]: m={m:.1f}  l={l:.4f}  O={O}  alpha={alpha:.4f}")
print("online result O/l      =", O / l)
print("match reference:", np.allclose(O / l, ref))
