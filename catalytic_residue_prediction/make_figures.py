import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import mannwhitneyu
import math

XRAY_PATH = os.path.join("evaluation", "train_xray")
MIXED_PATH = os.path.join("evaluation", "train_mixed")
ENTRIES_PATH = "../entries.idx"
FIGURE_DIR = "figures"
CPD_EMA_PDB_PATH = None # fill in with the location of the text files with pdb ids for CPD and EMA splits
CAT_PDB_PATH = os.path.join("data","splits")


######################
### UTIL FUNCTIONS ###
######################
def standard_error(vals):
    std = np.std(vals)
    err = std / (len(vals) ** 0.5)
    return err
    

def calculate_error(lines):
    vals = [float(line.split()[-1]) for line in lines]
    return standard_error(vals)


def get_auprc_per_class(path):
    auprc_per_class = dict()
    err_per_class = dict()
    auprc_list_per_class = dict()

    for test in ["test_xray", "test_nmr"]:
        auprc_per_class[test] = dict()
        err_per_class[test] = dict()

        auprc_list_per_class[test] = {1: [], 2: [], 3: [], 4: [], 5: [], 7: []}

        fname = test + "_ec_auprcs.txt"

        f = open(os.path.join(path, fname),"r").read().split("\n")[:-1]
        for line in f:
            ec = line.split()[2]
            c = int(ec.split(".")[0])
            auprc = float(line.split()[-1])
            auprc_list_per_class[test][c].append(auprc)
        for c in range(1,8):
            if c == 6:
                continue
            auprc_per_class[test][c] = np.mean(auprc_list_per_class[test][c])
            err_per_class[test][c] = standard_error(auprc_list_per_class[test][c])
  
    return auprc_per_class, err_per_class, auprc_list_per_class


def full_year(y):
    try:
        y = int(y)
    except:
        return 0
    else:
        if y < 22:
            return 2000 + y
        else:
            return 1900 + y



####################################
### PLOTTING FUNCTIONS: Figure 1 ###
####################################
def fig1():
    colordict = {"test_xray": {"train_xray": (200/255, 182/255, 210/255), "train_mixed": (106/255, 73/255, 142/255)},
                 "test_nmr": {"train_xray": (179/255, 212/255, 149/255), "train_mixed": (64/255, 146/255, 58/255)}}

    entries = pd.read_csv(ENTRIES_PATH, sep="\t", skiprows=2, names=["IDCODE", "HEADER", "ACCESSION DATE", "COMPOUND", "SOURCE", "AUTHOR LIST", "RESOLUTION", "EXPERIMENT TYPE"])
    years = [full_year(d[-2:]) for d in entries["ACCESSION DATE"].tolist()]
    entries["full year"] = years
    years_list = entries["full year"].unique()
    years_list.sort()
    fig, ax = plt.subplots()
    for y in years_list:
        if y < 1990 or y > 2020:
            continue
        types = entries.loc[entries["full year"] == y]["EXPERIMENT TYPE"].tolist()
        n_xray = len([x for x in types if "X-RAY DIFFRACTION" in x])
        n_nmr = len([x for x in types if "SOLUTION NMR" in x])
        n_em = len([x for x in types if "ELECTRON MICROSCOPY" in x])
        ax.bar(y, n_xray + n_nmr + n_em, width=1, color="gray")
        ax.bar(y, n_xray, width=1, color="darkviolet", alpha = 1.0)
        ax.bar(y, n_em, width=1, color="white", alpha = 1.0, label="_nolegend_")
        ax.bar(y, n_nmr, width=1, color="limegreen", alpha = 1.0)
        ax.bar(y, n_em, width=1, color="goldenrod", alpha = 0.8)
        
    plt.title("Experiment type of structures deposited in PDB, 1990-2020")
    plt.xlabel("Year")
    plt.ylabel("Number of structures")
    ax.legend(labels=["Total", "X-Ray Crystallography", "NMR", "Cryo-EM"])
    plt.savefig(os.path.join(FIGURE_DIR, "fig1.png"), dpi=400)
    plt.clf() 



###################################
### PLOTTING FUNCTIONS: DeepFRI ###
###################################
def res_pr_curve():
    colordict = {"test_xray": {"train_xray": (200/255, 182/255, 210/255), "train_mixed": (106/255, 73/255, 142/255)},
                 "test_nmr": {"train_xray": (179/255, 212/255, 149/255), "train_mixed": (64/255, 146/255, 58/255)}}
    xray_train_xray_test = pd.read_csv(os.path.join(XRAY_PATH, "test_xray_res_pr.csv"), index_col=0)
    mixed_train_xray_test = pd.read_csv(os.path.join(MIXED_PATH, "test_xray_res_pr.csv"), index_col=0)
    xray_train_nmr_test = pd.read_csv(os.path.join(XRAY_PATH, "test_nmr_res_pr.csv"), index_col=0)
    mixed_train_nmr_test = pd.read_csv(os.path.join(MIXED_PATH, "test_nmr_res_pr.csv"), index_col=0)

    xray_train_xray_test_auc = auc(xray_train_xray_test["Recall"], xray_train_xray_test["Precision"])
    mixed_train_xray_test_auc = auc(mixed_train_xray_test["Recall"], mixed_train_xray_test["Precision"])
    xray_train_nmr_test_auc = auc(xray_train_nmr_test["Recall"], xray_train_nmr_test["Precision"])
    mixed_train_nmr_test_auc = auc(mixed_train_nmr_test["Recall"], mixed_train_nmr_test["Precision"])

    plt.plot(xray_train_xray_test["Recall"], xray_train_xray_test["Precision"], label = "X-ray / Train: X-ray (AUPRC =" + str(xray_train_xray_test_auc) + ")", color=colordict["test_xray"]["train_xray"])
    plt.plot(mixed_train_xray_test["Recall"], mixed_train_xray_test["Precision"], label = "X-ray / Train: Mixed (AUPRC = " + str(mixed_train_xray_test_auc) + ")", color=colordict["test_xray"]["train_mixed"])
    plt.plot(xray_train_nmr_test["Recall"], xray_train_nmr_test["Precision"], label = "NMR / Train: X-ray (AUPRC = " + str(xray_train_nmr_test_auc) + ")", color=colordict["test_nmr"]["train_xray"])
    plt.plot(mixed_train_nmr_test["Recall"], mixed_train_nmr_test["Precision"], label = "NMR / Train: Mixed (AUPRC = " + str(mixed_train_nmr_test_auc) + ")", color=colordict["test_nmr"]["train_mixed"])


    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("DeepFRI catalytic residue identification")
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(FIGURE_DIR, "fig3b.png"),dpi=400)
    plt.clf()


def avg_auprc_bar_ec():
    colordict = {"xray": {XRAY_PATH: (200/255, 182/255, 210/255), MIXED_PATH: (106/255, 73/255, 142/255)},
                 "nmr": {XRAY_PATH: (179/255, 212/255, 149/255), MIXED_PATH: (64/255, 146/255, 58/255)}}
    bar_width = 0.33

    fig, ax = plt.subplots()
    x = 0
    for path in [XRAY_PATH, MIXED_PATH]:
        for test in ["test_xray", "test_nmr"]:
            fname = test + "_ec_auprcs.txt"
            fname = os.path.join(path, fname)
            f = open(fname, "r").read().split("\n")
            err = calculate_error(f[:-1])
            line = f[-1]
            auprc = float(line.split()[2])
            ax.bar(x, auprc, width = bar_width, color=colordict[test][path], yerr=err, capsize=10)

            x += bar_width
        x += bar_width
    plt.title("DeepFRI EC Number Prediction")
    plt.ylabel("Average AUPRC")
    plt.xticks([0, bar_width, 2.5 * bar_width, 3.5 * bar_width], ["Test: X-ray\nTrain: Mixed", "Test: X-ray\nTrain: X-ray", "Test: NMR\nTrain: Mixed", "Test: NMR\nTrain: X-ray"])
    plt.savefig(os.path.join(FIGURE_DIR, "fig4a.png"), dpi=400)
    plt.clf()


def avg_auprc_bar_ec_by_class():
    colordict = {"test_xray": {"train_xray": (200/255, 182/255, 210/255), "train_mixed": (106/255, 73/255, 142/255)},
                 "test_nmr": {"train_xray": (179/255, 212/255, 149/255), "train_mixed": (64/255, 146/255, 58/255)}}
    bar_width = 1
    cap_size = 4
    distdict = dict()
    auprc_per_class_xray, err_per_class_xray, distdict["train_xray"] = get_auprc_per_class(XRAY_PATH)
    auprc_per_class_mixed, err_per_class_mixed, distdict["train_mixed"] = get_auprc_per_class(MIXED_PATH)

    fig, ax = plt.subplots()
    X = 0
    for ec_class in range(1, 8):
        if ec_class == 6:
            continue
        for test in ["test_xray", "test_nmr"]:
            ax.bar(X, auprc_per_class_mixed[test][ec_class], width = bar_width, color=colordict[test]["train_mixed"], yerr=err_per_class_mixed[test][ec_class], capsize=cap_size)
            ax.bar(X + bar_width, auprc_per_class_xray[test][ec_class], width = bar_width, color=colordict[test]["train_xray"], yerr=err_per_class_xray[test][ec_class], capsize=cap_size)
            X += 2 * bar_width
        X += bar_width

        print("Mann-Whitney U test for Class " + str(ec_class))
        for pair in [(("test_xray", "train_xray"), ("test_xray", "train_mixed")),
                     (("test_nmr", "train_xray"), ("test_nmr", "train_mixed")),
                     (("test_xray", "train_xray"), ("test_nmr", "train_xray")),
                     (("test_xray", "train_mixed"), ("test_nmr", "train_mixed"))]:
            test1, train1 = pair[0]
            test2, train2 = pair[1]
            print(str(pair))
            print(mannwhitneyu(distdict[train1][test1][ec_class], distdict[train2][test2][ec_class]))
            print("----------")
        print("----------")

    plt.title("DeepFRI EC Number Prediction by Enzyme Class")
    plt.ylabel("Average AUPRC")
    ax.legend(labels=["X-ray / Train: Mixed", "X-ray / Train: X-ray", "NMR / Train: Mixed", "NMR / Train: X-ray"], bbox_to_anchor=(1.04,1), loc="upper left")
    ticks = ["Oxidoreductases","Transferases","Hydrolases","Lyases","Isomerases","Translocases"]
    plt.xticks(np.arange(bar_width * 1.5, 100, bar_width * 5)[:len(ticks)], ticks, rotation=40)
    plt.savefig(os.path.join(FIGURE_DIR, "fig4b.png"), bbox_inches="tight", dpi=400)
    plt.clf()




######################################
### PLOTTING FUNCTIONS: Supplement ###
######################################
def split_year_dists(train_list, val_list, test_lists, task, outf, lims, labels):
    colordict = {"X-RAY DIFFRACTION": "darkviolet",
                 "SOLUTION NMR": "limegreen",
                 "ELECTRON MICROSCOPY": "goldenrod"}

    entries = pd.read_csv(ENTRIES_PATH, sep="\t", skiprows=2, names=["IDCODE", "HEADER", "ACCESSION DATE", "COMPOUND", "SOURCE", "AUTHOR LIST", "RESOLUTION", "EXPERIMENT TYPE"])

    train = open(train_list,"r").read().split("\n")
    val = open(val_list,"r").read().split("\n")
    if type(test_lists) == list:
        test_xray = open(test_lists[0],"r").read().split("\n")
        test_non_xray = open(test_lists[1],"r").read().split("\n")
        test = test_xray + test_non_xray
    else:
        test = open(test_lists,"r").read().split("\n")

    counter = 1
    for name, split in [["training", train], ["validation", val], ["testing", test]]:
        split = [elem[:4].upper() for elem in split]
        df = entries.loc[entries["IDCODE"].isin(split)][["IDCODE","ACCESSION DATE","EXPERIMENT TYPE"]]
        years = [full_year(d[-2:]) for d in df["ACCESSION DATE"].tolist()]
        df["full year"] = years
        years_list = range(lims['xmin'], lims['xmax'])

        for method in ["X-RAY DIFFRACTION", "SOLUTION NMR", "ELECTRON MICROSCOPY"]:
            sub_df = df.loc[df["EXPERIMENT TYPE"] == method]
            if len(sub_df) == 0:
                counter += 3
                continue

            ax = plt.subplot(3, 3, counter)
            ax.set(xlim=(lims['xmin'], lims['xmax']))
            if counter < 4:
                plt.title(name)
            for y in years_list:
                prots = df.loc[df["full year"] == y]["IDCODE"].tolist()
                types = df.loc[df["full year"] == y]["EXPERIMENT TYPE"].tolist()
                examples_per_prot = dict()
                for p in split:
                    if p in examples_per_prot.keys():
                        examples_per_prot[p] += 1
                    else:
                         examples_per_prot[p] = 1
                n = sum([examples_per_prot[prots[i]] for i in range(len(prots)) if method in types[i]])
                ax.bar(y, n, width=1, color=colordict[method])
            counter += 3
        counter -= 8
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.suptitle(task + ": Distribution of deposition years per structure type")
    plt.text(labels["xlabelx"], labels["xlabely"], "Year", ha='center')
    plt.text(labels["ylabelx"], labels["ylabely"], "Number of structures", va='center', rotation='vertical')
    xray_patch = mpatches.Patch(color=colordict["X-RAY DIFFRACTION"], label="X-Ray Crystallography")
    nmr_patch = mpatches.Patch(color=colordict["SOLUTION NMR"], label="NMR")
    em_patch = mpatches.Patch(color=colordict["ELECTRON MICROSCOPY"], label="Cryo-EM")
    plt.legend(handles=[xray_patch, nmr_patch, em_patch], bbox_to_anchor=(1.04,3.9), loc="upper left")
    plt.savefig(outf, dpi=400, bbox_inches="tight")
    plt.clf() 


def split_res_dists(train_list, val_list, test_lists, task, outf, lims, labels):
    # no nmr because that doesnt have a resolution in the same way
    colordict = {"X-RAY DIFFRACTION": "darkviolet",
                 "ELECTRON MICROSCOPY": "goldenrod"}
    bin_width = 0.1

    entries = pd.read_csv(ENTRIES_PATH, sep="\t", skiprows=2, names=["IDCODE", "HEADER", "ACCESSION DATE", "COMPOUND", "SOURCE", "AUTHOR LIST", "RESOLUTION", "EXPERIMENT TYPE"])

    train = open(train_list,"r").read().split("\n")
    val = open(val_list,"r").read().split("\n")
    if type(test_lists) == list:
        test_xray = open(test_lists[0],"r").read().split("\n")
        test_non_xray = open(test_lists[1],"r").read().split("\n")
        test = test_xray + test_non_xray
    else:
        test = open(test_lists,"r").read().split("\n")

    counter = 1
    for name, split in [["training", train], ["validation", val], ["testing", test]]:
        split = [elem[:4].upper() for elem in split]
        df = entries.loc[entries["IDCODE"].isin(split)][["IDCODE","RESOLUTION","EXPERIMENT TYPE"]]

        for method in ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]:
            sub_df = df.loc[df["EXPERIMENT TYPE"] == method]
            if len(sub_df) == 0:
                counter += 3
                continue

            res_list = []
            for r in sub_df["RESOLUTION"]:
                try:
                    tmp = float(r)
                except:
                    spl = r.split(",")
                    if len(spl) > 1:
                        try:
                            l = [float(elem) for elem in spl]
                        except:
                            continue
                        else:
                            res_list.append(min(l))
                else:
                    res_list.append(tmp)

            bins = np.arange(0,math.ceil(max(res_list)), bin_width)
            bin_counts = dict()
            for r in res_list:
                for i in range(len(bins)):
                    if bins[i] > r:
                        if bins[i-1] in bin_counts.keys():
                            bin_counts[bins[i-1]] += 1
                        else:
                            bin_counts[bins[i-1]] = 1
                        break

            ax = plt.subplot(2, 3, counter)
            ax.set(xlim=(lims['xmin'], lims['xmax']))
            if counter < 4:
                plt.title(name)
            for b in bins:
                if b in bin_counts.keys():
                    ax.bar(b, bin_counts[b], width=bin_width, color=colordict[method])
            counter += 3
        counter -= 5
        
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.suptitle(task + ": Distribution of resolutions per structure type")
    plt.text(labels["xlabelx"], labels["xlabely"], "Resolution (Angstroms)", ha='center')
    plt.text(labels["ylabelx"], labels["ylabely"], "Number of structures", va='center', rotation='vertical')
    xray_patch = mpatches.Patch(color=colordict["X-RAY DIFFRACTION"], label="X-Ray Crystallography")
    em_patch = mpatches.Patch(color=colordict["ELECTRON MICROSCOPY"], label="Cryo-EM")
    plt.legend(handles=[xray_patch, em_patch], bbox_to_anchor=(1.04,2.28), loc="upper left")
    plt.savefig(outf, dpi=400, bbox_inches="tight")
    plt.clf() 


def split_res_dists_all():
    if CPD_EMA_PDB_PATH != None: # define this at top of file to plot!
        split_res_dists(os.path.join(CPD_EMA_PDB_PATH, "cpd_train_pdbs.txt"),
                        os.path.join(CPD_EMA_PDB_PATH, "cpd_val_pdbs.txt"),
                        os.path.join(CPD_EMA_PDB_PATH, "cpd_test_pdbs.txt"),
                        "Protein Sequence Design",
                        os.path.join(FIGURE_DIR, "figS7.png"),
                        {"xmin": 0,
                         "xmax": 4,
                         "ymin": 0,
                         "ymax": {"X-RAY DIFFRACTION": 1600, "ELECTRON MICROSCOPY": 10}},
                        {"xlabelx": -3.5,
                         "xlabely": -0.5,
                         "ylabelx": -13,
                         "ylabely": 2.5})

        split_res_dists(os.path.join(CPD_EMA_PDB_PATH, "ema_train_pdbs.txt"),
                        os.path.join(CPD_EMA_PDB_PATH, "ema_val_pdbs.txt"),
                        os.path.join(CPD_EMA_PDB_PATH, "ema_test_pdbs.txt"),
                        "Estimation of Model Accuracy",
                        os.path.join(FIGURE_DIR, "figS8.png"),
                        {"xmin": 0,
                         "xmax": 4,
                         "ymin": 0,
                         "ymax": {"X-RAY DIFFRACTION": 35, "ELECTRON MICROSCOPY": 2}},
                        {"xlabelx": -2.7,
                         "xlabely": -0.3,
                         "ylabelx": -12.5,
                         "ylabely": 1})

    split_res_dists(os.path.join(CAT_PDB_PATH, "train_mixed.txt"),
                    os.path.join(CAT_PDB_PATH, "val_mixed.txt"),
                    [os.path.join(CAT_PDB_PATH, "test_xray.txt"),os.path.join(CAT_PDB_PATH, "test_nmr.txt")],
                    "Catalytic Residue Prediction",
                    os.path.join(FIGURE_DIR, "figS9.png"),
                    {"xmin": 0,
                     "xmax": 20,
                     "ymin": 0,
                     "ymax": {"X-RAY DIFFRACTION": 2500, "ELECTRON MICROSCOPY": 5}},
                    {"xlabelx": -17,
                     "xlabely": -1,
                     "ylabelx": -69,
                     "ylabely": 3})


def split_year_dists_all():
    if CPD_EMA_PDB_PATH != None: # define this at top of file to plot!
        split_year_dists(os.path.join(CPD_EMA_PDB_PATH, "cpd_train_pdbs.txt"),
                         os.path.join(CPD_EMA_PDB_PATH, "cpd_val_pdbs.txt"),
                         os.path.join(CPD_EMA_PDB_PATH, "cpd_test_pdbs.txt"),
                         "Protein Sequence Design",
                         os.path.join(FIGURE_DIR, "figS4.png"),
                         {"xmin": 1975,
                          "xmax": 2020,
                          "ymin": 0,
                          "ymax": {"X-RAY DIFFRACTION": 1000, "SOLUTION NMR": 320, "ELECTRON MICROSCOPY": 40}},
                         {"xlabelx": 1932,
                          "xlabely": -3,
                          "ylabelx": 1817,
                          "ylabely": 10})

        split_year_dists(os.path.join(CPD_EMA_PDB_PATH, "ema_train_pdbs.txt"),
                         os.path.join(CPD_EMA_PDB_PATH, "ema_val_pdbs.txt"),
                         os.path.join(CPD_EMA_PDB_PATH, "ema_test_pdbs.txt"),
                         "Estimation of Model Accuracy",
                         os.path.join(FIGURE_DIR, "figS5.png"),
                         {"xmin": 2005,
                          "xmax": 2021,
                          "ymin": 0,
                          "ymax": {"X-RAY DIFFRACTION": 80, "SOLUTION NMR": 20, "ELECTRON MICROSCOPY": 4}},
                         {"xlabelx": 1990,
                          "xlabely": -0.4,
                          "ylabelx": 1952,
                          "ylabely": 2})
    
    split_year_dists(os.path.join(CAT_PDB_PATH, "train_mixed.txt"),
                     os.path.join(CAT_PDB_PATH, "val_mixed.txt"),
                     [os.path.join(CAT_PDB_PATH, "test_xray.txt"),os.path.join(CAT_PDB_PATH, "test_nmr.txt")],
                     "Catalytic Residue Prediction",
                     os.path.join(FIGURE_DIR, "figS6.png"),
                     {"xmin": 1970,
                      "xmax": 2015,
                      "ymin": 0,
                      "ymax": {"X-RAY DIFFRACTION": 1700, "SOLUTION NMR": 15, "ELECTRON MICROSCOPY": 18}},
                     {"xlabelx": 1930,
                      "xlabely": -4,
                      "ylabelx": 1822,
                      "ylabely": 20})


if __name__ == "__main__":
    fig1()
    res_pr_curve() # fig 3b
    avg_auprc_bar_ec() # fig 4a
    avg_auprc_bar_ec_by_class() # fig 4b
    split_year_dists_all() # figs S4-6
    split_res_dists_all() # figs S7-9
