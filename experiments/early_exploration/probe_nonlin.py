import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import torch, numpy as np, warnings
warnings.filterwarnings("ignore")
from aoc import AOCCell, MatrixConnectivityType
cell = AOCCell.from_parameters([48,48], connectivity=MatrixConnectivityType.FEEDBACK,
                               normalise_matrix=False, alpha=0.5)
z = torch.linspace(-2.6, 2.6, 5201, dtype=torch.float64)
t = cell._aoc_tanh(z.float()).double()
lp, ln = cell._uled_nonlinearity(t.float())
lp, ln = lp.double(), ln.double()
def d(y): return torch.gradient(y, spacing=(z,))[0]
sp, sn = d(lp), d(ln)
i0 = torch.argmin(z.abs())
print(f"tanh output range: [{t.min():.4f}, {t.max():.4f}] V")
print(f"LED(+) at z=0: {lp[i0]:.5f}, LED(-) at z=0: {ln[i0]:.5f}  (DC offsets)")
pk_p, pk_n = sp.max(), sn.max()
print(f"peak slope: pos {pk_p:.4f} at z={z[sp.argmax()]:.3f} V ; neg {pk_n:.4f} at z={z[sn.argmax()]:.3f} V")
for zz in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 2.0]:
    j = torch.argmin((z-zz).abs())
    print(f" z={zz:+.2f}V  slope/peak pos={sp[j]/pk_p:.4f} neg={sn[j]/pk_n:.4f}  ratio pos/neg={sp[j]/sn[j]:.3f}")
# asymmetry: slope at +z vs -z
for zz in [0.2,0.4,0.6]:
    j1 = torch.argmin((z-zz).abs()); j2 = torch.argmin((z+zz).abs())
    print(f" asym z=±{zz}: pos slope +z/-z = {sp[j1]/sp[j2]:.3f}")
np.savez("nonlin.npz", z=z.numpy(), lp=lp.numpy(), ln=ln.numpy(), sp=sp.numpy(), sn=sn.numpy())
print("deq_input_min_max", cell.hardware_parameters.deq_input_min_max)
