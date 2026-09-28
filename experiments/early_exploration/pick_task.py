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
for a, b in [(3,8),(3,5),(5,8),(4,9),(1,7),(7,9),(2,3)]:
    m = (y==a)|(y==b); Xa, ya = X[m], (y[m]==b).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, random_state=0, stratify=ya)
    p = PCA(8).fit(Xtr); Ztr, Zte = p.transform(Xtr), p.transform(Xte)
    lr = LogisticRegression(max_iter=2000).fit(Ztr,ytr).score(Zte,yte)
    mlp = np.mean([MLPClassifier((16,),max_iter=3000,random_state=s).fit(Ztr,ytr).score(Zte,yte) for s in range(3)])
    print(f"{a} vs {b}: n={m.sum()}  LR={lr:.3f}  MLP16={mlp:.3f}")
