import pandas as pd
import sys
from sklearn.metrics import precision_recall_curve, auc
import os
import numpy as np

def binarize_true(prot_order, true, ec):
    y_test = []
    for prot in prot_order:
        true_ec = true.loc[true["PDB chain"] == prot]["EC numbers"].tolist()[0]
        true_ec = true_ec.split(",")
        if ec in true_ec:
            y_test.append(1)
        else:
            y_test.append(0)
    return y_test


def binarize_pred(prot_order, pred, ec):
    y_pred = []
    for prot in prot_order:
        pred_ec = pred.loc[pred["Protein"] == prot]
        pred_ec = pred_ec.loc[pred_ec["GO_term/EC_number"] == ec]
        if len(pred_ec) == 1:
            y_pred.append(pred_ec["Score"].tolist()[0])
        elif len(pred_ec) == 0:
            y_pred.append(0)
        else:
            print("something unexpected happened with " + ec)
            break
    return y_pred


if __name__ == "__main__":
    test_split = sys.argv[1]
    model_name = sys.argv[2]

    outf = os.path.join("evaluation", model_name, test_split + "_ec_auprcs.txt")
    csv_dir = os.path.join("evaluation", model_name, "ec_csvs", test_split)

    true = pd.read_csv("data/annot_files/" + test_split + "_annot.tsv", skiprows=4, names=["PDB chain", "EC numbers"], sep="\t")
    pred = pd.read_csv("test_outputs/" + model_name + "_" + test_split + "_predictions.csv", skiprows=1)
    test_ecs = open("data/annot_files/" + test_split + "_annot.tsv","r").read().split("\n")[1].split("\t")
    train_ecs = open("data/annot_files/train_val_annot.tsv","r").read().split("\n")[1].split("\t")
    prot_order = true["PDB chain"].tolist()


    f = open(outf, "w")
    auprc_sum = 0
    auprc_total = 0

    for ec in test_ecs:
        if not ec in train_ecs:
            continue
        y_test = binarize_true(prot_order, true, ec)
        y_score = binarize_pred(prot_order, pred, ec)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        auprc = auc(recall, precision)
        f.write("AUPRC for " + ec + ": " + str(auprc) + "\n")
        print("AUPRC for " + ec + ": " + str(auprc))
        auprc_sum += auprc
        auprc_total += 1

        results_dict = {"PDB ID - Chain": prot_order, "True label": y_test, "Predicted label": y_score}
        results_df = pd.DataFrame(results_dict)
        results_df.to_csv(os.path.join(csv_dir, test_split + "_" + ec + "_ec_preds.csv"))

        pr_dict = {'Precision': precision, 'Recall': recall, 'Thresholds': np.concatenate((thresholds, np.array([1])))}
        pr_df = pd.DataFrame(pr_dict)
        pr_df.to_csv(os.path.join(csv_dir, test_split + "_" + ec + "_ec_pr.csv"))

    f.write("Overall AUPRC: " + str(auprc_sum / auprc_total))
    print("Overall AUPRC: " + str(auprc_sum / auprc_total))
    f.close()
