"""Infer the Coverage denominator from the 1st-place official score.

official(only_ngrams, private) = 0.55569 = 0.8 * DCG_test + 0.2 * U_test / D
U_test is exact (we reproduce their test predictions); DCG_test is estimated from validation,
corrected for the larger corpus available at test time (measured by shrinking the val corpus).
"""
import numpy as np
from common import WORK

p = np.load(WORK / "winner_pred_test.npy")
U = len(np.unique(p))
print("U_test =", U, "slots =", p.size)
for dcg in [0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68]:
    cov = (0.55569 - 0.8 * dcg) / 0.2
    print(f"if test DCG={dcg:.2f}: coverage={cov:.4f} -> D={U / cov:,.0f}" if cov > 0 else f"if test DCG={dcg}: impossible")
print("reference sizes: slots=715,320  meta_song=1,030,712  played songs=773,314")
