import torch
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from statistics import mean
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
LOGGER = logging.getLogger('MetricLogger')


def standard_metrics(probs, labels, *args, **kwargs):
    if len(probs.shape) == 1 and torch.all(torch.logical_or(labels == 0, labels == 1)):
        return standard_metrics_binary(probs, labels, *args, **kwargs)
    else:
        return standard_metrics_multiclass(probs, labels, *args, **kwargs)


def standard_metrics_binary(probs, labels, threshold=0.5, add_aucroc=True, add_optimal_acc=False, **kwargs):
    """
    Given predicted probabilities and labels, returns the standard metrics of accuracy, recall, precision, F1 score and AUCROC.
    The threshold, above which data points are considered to be classified as class 1, can also be adjusted.
    Returned values are floats (no tensors) in the dictionary.
    Probabilities and labels are expected to be pytorch tensors.
    """
    assert torch.all(torch.logical_and(probs <= 1.0, probs >= 0.0)), "Probabilities must be between 0 and 1, but are as follows: " + str(probs)
    assert torch.all(torch.logical_or(labels == 0, labels == 1)), "Labels must be binary (0 or 1), but are as follows: " + str(labels)
    if torch.all(torch.logical_or(probs == 0, probs == 1)):
        LOGGER.warning("Standard metrics received discrete predictions as probabilities, but expects probabilities between 0.0 and 1.0. Are you sure the inputs are correct?")

    preds = (probs > threshold).long()
    eval_dict = get_TFPN_dict(preds, labels, true_label=1, as_float=True)
    metrics = dict()
    metrics["accuracy"] = (eval_dict["TP"] + eval_dict["TN"]) / preds.shape[0]
    metrics["recall"] = eval_dict["TP"] / (eval_dict["TP"] + eval_dict["FN"]).clamp(min=1e-4)
    metrics["precision"] = eval_dict["TP"] / (eval_dict["TP"] + eval_dict["FP"]).clamp(min=1e-4)
    if metrics["recall"] == 0.0 or metrics["precision"] == 0.0:
        metrics["F1"] = 0.0
    else:
        metrics["F1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])

    if add_aucroc:
        metrics["aucroc"] = aucroc(probs, labels)

    if add_optimal_acc:
        threshold = find_optimal_threshold(probs, labels, metric="accuracy", show_plot=False)
        metrics["optimal_threshold"] = threshold
        metrics["optimal_accuracy"] = standard_metrics(probs, labels, threshold=threshold, add_aucroc=False, add_optimal_acc=False)["accuracy"]

    metrics = {key: metrics[key].item() if isinstance(metrics[key], torch.Tensor) else metrics[key] for key in metrics}
    return metrics


def standard_metrics_multiclass(probs, labels, **kwargs):
    assert len(probs.shape) == 2, "Probabilities need to be given for each class."
    preds = probs.argmax(dim=-1)
    # print("Unique labels", torch.unique(labels))
    # print("Unique preds", torch.unique(preds))
    # print("Shape", probs.shape)
    eval_dict = [get_TFPN_dict(preds, labels, true_label=i, as_float=True) for i in range(probs.shape[1])]
    metrics = dict()
    metrics["accuracy"] = (preds == labels).float().mean()
    recalls = [d["TP"] / (d["TP"]+d["FN"]).clamp(min=1e-4) for d in eval_dict]
    precisions = [d["TP"] / (d["TP"]+d["FP"]).clamp(min=1e-4) for d in eval_dict]
    f1_scores = [(2*r*p/(r+p) if (r+p)>0.0 else 0.0) for r, p in zip(recalls, precisions)]
    metrics.update({
        "recall": sum(recalls)/len(recalls),
        "precision": sum(precisions)/len(precisions),
        "F1": sum(f1_scores)/len(f1_scores),
        "aucroc": -1.0,
        "optimal_threshold": -1.0,
        "optimal_accuracy": -1.0
    })

    metrics = {key: metrics[key].item() if isinstance(metrics[key], torch.Tensor) else metrics[key] for key in metrics}
    return metrics


def get_TFPN_dict(preds, labels, true_label=1, as_float=False):
    """
    Given predictions and labels, returns a dictionary with TP, TN, FP and FN for a given class label.
    """
    eval_dict = dict()
    eval_dict["TP"] = torch.logical_and(preds == true_label, preds == labels)
    eval_dict["TN"] = torch.logical_and(preds != true_label, preds == labels)
    eval_dict["FP"] = torch.logical_and(preds == true_label, preds != labels)
    eval_dict["FN"] = torch.logical_and(preds != true_label, preds != labels)
    eval_dict = {key: eval_dict[key].long().sum() for key in eval_dict}
    if as_float:
        eval_dict = {key: eval_dict[key].float() for key in eval_dict}
    return eval_dict


def find_optimal_threshold(probs, labels, metric="accuracy", show_plot=False):
    """
    Given predicted probabilities and labels, returns the optimal threshold to use for the binary classification.
    It is conditioned on a metric ("accuracy", "F1", ...). For interpretability, the score over thresholds can
    also be plotted with the option "show_plot". Probabilities and labels are expected to be pytorch tensors.
    """
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    def_dict = standard_metrics(probs, labels)
    assert metric in def_dict, "Metric needs to be a key of the dict standard_metrics returns, but is not. " + \
                               "Given metric: \"%s\", possible metrics: \"%s\"" % (metric, str(def_dict.keys()))
    
    scores = []
    def add_score(thresh):
        scores.append((thresh, standard_metrics_binary(probs, labels, thresh, add_aucroc=False)[metric]))
    add_score(0.0)
    pos_thresh = torch.sort(probs)[0].detach().cpu().numpy()
    for t in pos_thresh:
        add_score(t)
    add_score(1.0)
    
    argmax = np.array([s[1] for s in scores]).argmax()
    if argmax != len(scores)-1 and argmax != 0:
        best_threshold = (scores[argmax][0] + scores[argmax+1][0]) / 2
    else:
        best_threshold = scores[argmax][0]
    reproduced_score = standard_metrics_binary(probs, labels, best_threshold)[metric] 
    if reproduced_score != scores[argmax][1]:
        LOGGER.warning("Internal error. Was not able to reproduce best threshold score.\nOriginal score: %f\nReproduced score: %f" % (scores[argmax][1], reproduced_score))
    
    if show_plot:
        x_axis, y_axis = [], []
        for i in range(len(scores)-1):
            x_axis.extend([scores[i][0], scores[i+1][0]])
            y_axis.extend([scores[i][1], scores[i][1]])
        sns.set()
        plt.plot(x_axis, y_axis, color="C0")
        plt.fill_between(x_axis, [0 for _ in y_axis], y_axis, color="C0", alpha=0.3)
        plt.plot([best_threshold, best_threshold, 0.0], [0.0, scores[argmax][1], scores[argmax][1]], '--', color="C2", linewidth=2)
        plt.plot([best_threshold], [scores[argmax][1]], '*', markersize=14, markerfacecolor="C2", markeredgecolor="#000000")
        plt.title("%s over thresholds" % metric.capitalize())
        plt.xlabel("Thresholds")
        plt.ylabel(metric.capitalize())
        plt.ylim([0.0, scores[argmax][1]*1.1])
        plt.xlim([0.0, 1.0])
        plt.show()
    
    return best_threshold


def aucroc(probs, labels):
    """
    Given predicted probabilities and labels, returns the AUCROC score used in the Facebook Meme Challenge.
    Inputs are expected to be pytorch tensors (can be cuda or cpu)
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    assert np.all(np.logical_and(probs <= 1.0, probs >= 0.0)), "Probabilities must be between 0 and 1"
    assert np.all(np.logical_or(labels == 0, labels == 1)), "Labels must be binary (0 or 1)"
    if not (np.any(labels == 0) and np.any(labels == 1)):
        LOGGER.warning("ROC AUC calculation got only one label. Score not defined here, setting it to 0.")
        return 0.0
    
    aucroc_score = roc_auc_score(y_true=labels, y_score=probs, average='macro')
    return aucroc_score


if __name__ == '__main__':
    # probs = torch.rand(size=(1000,))
    # labels = (probs > torch.rand(size=probs.shape)).long()
    # t = find_optimal_threshold(probs, labels, metric="accuracy", show_plot=True)
    # print("Optimal threshold: %5.3f" % t)
    # print("Metrics", standard_metrics(probs, labels))

    num_classes = 4
    probs = torch.randn(size=(1000,num_classes))
    probs = F.softmax(probs, dim=-1)
    labels = torch.multinomial(probs, num_samples=1).squeeze() * 0
    print("Metrics", standard_metrics(probs, labels))