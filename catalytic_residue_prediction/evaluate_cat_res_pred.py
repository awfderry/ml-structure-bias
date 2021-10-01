import sys
from sklearn.metrics import precision_recall_curve, auc
import json
import numpy as np
import pandas as pd
from Bio import PDB
import os


def get_preds(centers, pred_json, ec_labels, pdbparser, split):
    y_pred = []
    pdbs = []
    chains = []
    ns = []
    for i, line in enumerate(centers):
        chain, n = line.split("\t")
        if not chain in pred_json.keys():
            print(chain + " was not tested, skipping...")
            continue
        pdb_id = chain[:4]
        pdb_file = "data/" + split + "/pdbs/" + pdb_id + ".pdb"
        start = get_start_n(pdbparser, pdb_id, pdb_file, chain[-1])
        print(chain, n, start)
        res_n = int(n)
        n = res_n - start
        if n >= len(pred_json[chain]['sequence']):
            # something went wrong with the negative sampling...
            # proceeding will lead to an index out of bounds error
            continue
        true_ecs = ec_labels.loc[ec_labels["PDB chain"] == chain]["EC numbers"].tolist()
        if len(true_ecs) == 0:
            # this happens if there isnt an EC label
            # in CSA there is a label for the catalytic residue
            # but we don't know what EC number to look under
            # so we will skip the protein
            print("skipping " + chain)
            continue
        true_ecs = true_ecs[0].split(",")
        for ec in true_ecs:
            if ec in pred_json[chain]['GO_ids']:
                ec_idx = pred_json[chain]['GO_ids'].index(ec)
                y_pred.append(pred_json[chain]['saliency_maps'][ec_idx][n])
                pdbs.append(pdb_id)
                chains.append(chain[-1])
                ns.append(res_n)
            else:
                short_test_ecs = [make_shorter(ec) for ec in pred_json[chain]['GO_ids']]
                short_target_ec = make_shorter(ec)
                if short_target_ec in short_test_ecs:
                    ec_idx = short_test_ecs.index(short_target_ec)
                    y_pred.append(pred_json[chain]['saliency_maps'][ec_idx][n])
                    pdbs.append(pdb_id)
                    chains.append(chain[-1])
                    ns.append(res_n)
    return np.array(y_pred), np.array(pdbs), np.array(chains), np.array(ns)


def make_shorter(ec):
    return ".".join(ec.split(".")[:-1]) 


def get_start_n(pdbparser, pdb_id, infile, chain):
    structure = pdbparser.get_structure(pdb_id, infile)
    try:
        c = structure[0][chain]
    except:
        print("ERROR: could not parse pdb " + pdb_id)
        return None
    else:
        start = next(c.get_residues())
        return start.get_full_id()[3][1]


if __name__ == "__main__":
    split = sys.argv[1]
    model_name = sys.argv[2]

    prefix = os.path.join("evaluation", model_name, split + "_res_")
    preds_csv = prefix + "preds.csv"
    pr_csv = prefix + "pr.csv"
    outfile = open(prefix + "auprc.txt", "w")


    pos = open("data/" + split + "/centers_" + split + "_pos.tsv","r").read().split("\n")[:-1]
    neg = open("data/" + split + "/centers_" + split + "_neg.tsv","r").read().split("\n")[:-1]
    pred = json.load(open("test_outputs/" + model_name + "_" + split + "_saliency_maps.json","r"))
    ec_labels = pd.read_csv("data/annot_files/" + split + "_annot.tsv", skiprows=4, names=["PDB chain", "EC numbers"], sep="\t")

    pdbparser = PDB.PDBParser()

    print("getting predictions...")
    y_pred_pos, pdbs_pos, chains_pos, ns_pos = get_preds(pos, pred, ec_labels, pdbparser, split)
    y_pred_neg, pdbs_neg, chains_neg, ns_neg = get_preds(neg, pred, ec_labels, pdbparser, split)
    y_pred = np.concatenate((y_pred_pos, y_pred_neg))
    pdbs = np.concatenate((pdbs_pos, pdbs_neg))
    chains = np.concatenate((chains_pos, chains_neg))
    ns = np.concatenate((ns_pos, ns_neg))

    y_test_pos = np.ones((y_pred_pos.shape[0],))
    y_test_neg = np.zeros((y_pred_neg.shape[0],))
    y_test = np.concatenate((y_test_pos, y_test_neg))

    outfile.write("number of positives to evaluate: " + str(y_pred_pos.shape[0]) + "\n")
    outfile.write("number of negatives to evaluate: " + str(y_pred_neg.shape[0]) + "\n")

    print("computing auprc...")
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    auprc = auc(recall, precision)

    outfile.write("Overall AUPRC: " + str(auprc))
    outfile.close()

    results_dict = {"PDB ID": pdbs, "Chain": chains, "Residue Number": ns, "True label": y_test, "Predicted label": y_pred}
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(preds_csv)

    pr_dict = {'Precision': precision, 'Recall': recall, 'Thresholds': np.concatenate((thresholds, np.array([1])))}
    pr_df = pd.DataFrame(pr_dict)
    pr_df.to_csv(pr_csv)
