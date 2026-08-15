from __future__ import annotations
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np

def evaluate(y_true, scores, threshold: float) -> dict:
    y=np.asarray(y_true,dtype=int); s=np.asarray(scores,dtype=float); pred=(s>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    sensitivity=float(tp/(tp+fn)) if tp+fn else None
    auc=float(roc_auc_score(y,s)) if len(set(y.tolist()))>1 else None
    return {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"sensitivity":sensitivity,"roc_auc":auc,"threshold":float(threshold)}
