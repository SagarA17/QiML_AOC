import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.join(_ROOT, "src", "qiml_aoc")); _os.chdir(_os.path.join(_ROOT, "results"))
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
tasks = {"even vs odd": (y%2), "0-4 vs 5-9": (y>=5).astype(int),
         "{3,5,8} vs rest": np.isin(y,[3,5,8]).astype(int)}
for name, lab in tasks.items():
    for npc in [6, 8]:
        Xtr, Xte, ytr, yte = train_test_split(X, lab, test_size=0.3, random_state=0, stratify=lab)
        p = PCA(npc).fit(Xtr); Ztr, Zte = p.transform(Xtr), p.transform(Xte)
        lr = LogisticRegression(max_iter=3000).fit(Ztr,ytr).score(Zte,yte)
        mlp = np.mean([MLPClassifier((32,),max_iter=4000,random_state=s).fit(Ztr,ytr).score(Zte,yte) for s in range(3)])
        print(f"{name:18s} PCA{npc}: ntest={len(yte)} LR={lr:.3f} MLP32={mlp:.3f}")
