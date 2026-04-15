import editdistance
from collections import defaultdict


def cer_per_perturbation_type(y_test, predictions, perturbation_type_test):
    """
    Calculate mean Character Error Rate (CER) grouped by perturbation type.

    Args:
        y_test: list of ground truth strings
        predictions: list of predicted strings
        perturbation_type_test: array where each value is None (clean) or a perturbation type string

    Returns:
        dict mapping perturbation type name to mean CER for that group
        (None perturbation types are grouped under 'clean')
    """
    groups = defaultdict(list)

    for gt, pred, ptype in zip(y_test, predictions, perturbation_type_test):
        if len(gt) == 0:
            continue
        cer = editdistance.eval(pred, gt) / len(gt)
        key = 'clean' if ptype is None else ptype
        groups[key].append(cer)

    return {key: sum(cers) / len(cers) for key, cers in groups.items()}
