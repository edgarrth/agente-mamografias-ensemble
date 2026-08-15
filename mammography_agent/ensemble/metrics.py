from __future__ import annotations
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np

def evaluate(y_true, scores, threshold: float) -> dict:
    y=np.asarray(y_true,dtype=int); s=np.asarray(scores,dtype=float); pred=(s>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    positives=int((y==1).sum()); negatives=int((y==0).sum())
    sensitivity=float(tp/(tp+fn)) if positives else None
    sensitivity_reason=None if positives else "Sensitivity requires at least one malignant (ground_truth=1) study."
    auc=float(roc_auc_score(y,s)) if positives and negatives else None
    auc_reason=None if auc is not None else "ROC-AUC requires both benign (0) and malignant (1) ground-truth classes."
    return {
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "sensitivity":sensitivity,"sensitivity_unavailable_reason":sensitivity_reason,
        "roc_auc":auc,"roc_auc_unavailable_reason":auc_reason,
        "threshold":float(threshold),
    }
