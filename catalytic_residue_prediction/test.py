from deepfri.Predictor import Predictor
import sys
import os

for train in ["train_xray", "train_mixed"]:
    predictor = Predictor(train)
    for test in ["test_xray", "test_nmr"]:
        print("Evaluating " + train + " model on " + test + " data...")

        cmap_dir = os.path.join("data",test,"cmaps")
        files = [ os.path.join(cmap_dir,f) for f in os.listdir(cmap_dir) if os.path.isfile(os.path.join(cmap_dir,f))]
        files = [t for t in files if t[-4:] == ".npz"]

        for i, cmap in enumerate(files):
            if i % 100 == 0:
                print(str(i) + " of " + str(len(files)) + "...")
            predictor.predict(cmap, chain=cmap[-10:-4])

        # save predictions for EC number prediction
        print("Saving predictions...")
        output_name = model_name + "_" + test
        predictor.export_csv("test_outputs/" + output_name + "_predictions.csv", verbose = False)

        # save saliency maps for catalytic residue prediction
        print("Computing saliency maps...")
        predictor.compute_GradCAM()
        predictor.save_GradCAM("test_outputs/" + output_name + "_saliency_maps.json")

print("done")
