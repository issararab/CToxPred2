import os
import joblib
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.QED import properties

import utils.global_variable as glv
from CToxPred2.utils import (
    compute_fingerprint_features,
    compute_descriptor_features,
)
from CToxPred2.hERG_model import hERGClassifier
from CToxPred2.nav15_model import Nav15Classifier
from CToxPred2.cav12_model import Cav12Classifier


def set_window_center(window, width, height):
    w_s = window.winfo_screenwidth()
    h_s = window.winfo_screenheight()
    x_co = (w_s - width) / 2
    y_co = (h_s - height) / 2 - 50
    window.geometry("%dx%d+%d+%d" % (width, height, x_co, y_co))
    window.minsize(width, height)


def get_screen_size(window):
    return window.winfo_screenwidth(), window.winfo_screenheight()


def get_window_size(window):
    return window.winfo_reqwidth(), window.winfo_reqheight()


def treeview_sort_column(tv, col, reverse):
    l = [(tv.set(k, col), k) for k in tv.get_children("")]
    l.sort(reverse=reverse)
    # rearrange items in sorted positions
    for index, (val, k) in enumerate(l):
        tv.move(k, "", index)
    tv.heading(col, command=lambda: treeview_sort_column(tv, col, not reverse))


def split_list_to_columns(row):
    values = list(row["prop"])
    new_columns = pd.Series(values)
    new_columns = new_columns.rename(
        {
            0: "MW",
            1: "AlogP",
            2: "HBA",
            3: "HBD",
            4: "MPSA",
            5: "ROTB",
            6: "AROMS",
            7: "ALERTS",
        }
    )
    return new_columns


def resolve_2d_structure(smiles_string):
    # Create an RDKit molecule object and standardize it
    s = Standardizer()
    mol_object = Chem.MolFromSmiles(smiles_string, sanitize=True)
    smol_object = s.standardize(mol_object)

    # Generate an image of the molecule
    img = Draw.MolToImage(smol_object, size=(300, 300))

    return img


# CToxPred2 predictions
def _generate_predictions_ssl(smiles_list: List[str]) -> Dict[str, any]:
    """
    Generates predictions for hERG, Nav1.5, and Cav1.2 targets based on the
    provided list of SMILES. The model used was trained using a
    semi-supervised learning startegy.

    This function processes the input SMILES list and computes fingerprint
    and descriptor features for each compound. Then, it loads pre-trained
    models for hERG, Nav1.5, and Cav1.2 targets,  and predicts the activity
    of each compound for these targets using the
    respective models. The model uses  majority voting of random-forest to
    compute the model uncertainty. The 'hERG', 'Nav1.5', and 'Cav1.2'
    columns contain the binary predictions (0 or 1) for each target,
    representing non-toxic (negative class) or toxic (positive class) compounds, respectively.

    Parameters:
        smiles_list: List[str]
            A list containing SMILES strings of chemical compounds.

    Returns:
        Dict[str, any]:
            The function returns the predictions as a dictionary.
    """
    # Compute features
    glv.LOGGER.info("Calculate Features!")
    fingerprints = compute_fingerprint_features(smiles_list)
    descriptors = compute_descriptor_features(smiles_list)
    # Load data pre-processing pipeline
    path = [
        "CToxPred2",
        "models",
        "decriptors_preprocessing",
        "global_preprocessing_pipeline.sav",
    ]
    descriptors_transformation_pipeline = joblib.load(os.path.join(*path))
    descriptors = descriptors_transformation_pipeline.transform(descriptors)
    molecular_features = np.concatenate((fingerprints, descriptors), axis=1)
    # Process hERG
    glv.LOGGER.info("Predict hERG toxicity!")
    path = ["CToxPred2", "models", "random_forest", "hERG", "_ssl_herg_model.joblib"]
    herg_model = joblib.load(os.path.join(*path))
    proba_predictions = herg_model.predict_proba(molecular_features)
    hERG_predictions, hERG_confidence_predictions = np.argmax(
        proba_predictions, axis=1
    ), np.max(proba_predictions, axis=1)
    # Process Nav1.5
    glv.LOGGER.info("Predict Nav1.5 toxicity!")
    path = ["CToxPred2", "models", "random_forest", "Nav1.5", "_ssl_nav_model.joblib"]
    nav_model = joblib.load(os.path.join(*path))
    proba_predictions = nav_model.predict_proba(molecular_features)
    nav15_predictions, nav15_confidence_predictions = np.argmax(
        proba_predictions, axis=1
    ), np.max(proba_predictions, axis=1)
    # Process Cav1.2
    glv.LOGGER.info("Predict Cav1.2 toxicity!")
    path = ["CToxPred2", "models", "random_forest", "Cav1.2", "_ssl_cav_model.joblib"]
    herg_model = joblib.load(os.path.join(*path))
    proba_predictions = herg_model.predict_proba(molecular_features)
    cav12_predictions, cav12_confidence_predictions = np.argmax(
        proba_predictions, axis=1
    ), np.max(proba_predictions, axis=1)

    # Data formating
    hERG_predictions, hERG_confidence_predictions = zip(
        *[
            (prediction.item(), "{:.1%}".format(confidence.item()))
            for prediction, confidence in zip(
                hERG_predictions, hERG_confidence_predictions
            )
        ]
    )

    nav15_predictions, nav15_confidence_predictions = zip(
        *[
            (prediction.item(), "{:.1%}".format(confidence.item()))
            for prediction, confidence in zip(
                nav15_predictions, nav15_confidence_predictions
            )
        ]
    )

    cav12_predictions, cav12_confidence_predictions = zip(
        *[
            (prediction.item(), "{:.1%}".format(confidence.item()))
            for prediction, confidence in zip(
                cav12_predictions, cav12_confidence_predictions
            )
        ]
    )

    return {
        "hERG": list(hERG_predictions),
        "hERG_confidence": list(hERG_confidence_predictions),
        "Nav1.5": list(nav15_predictions),
        "Nav1.5_confidence": list(nav15_confidence_predictions),
        "Cav1.2": list(cav12_predictions),
        "Cav1.2_confidence": list(cav12_confidence_predictions),
    }


def _generate_predictions_sl(smiles_list: List[str]) -> Dict[str, any]:
    """
    Generates predictions for hERG, Nav1.5, and Cav1.2 targets based on the
    provided list of SMILES. The model used was trained using a supervised
    learning startegy.

    This function processes the input SMILES list and computes fingerprint and descriptor features for each compound.
    Then, it loads pre-trained models for hERG, Nav1.5, and Cav1.2 targets, and predicts the activity of each compound
    for these targets using the respective models. The function uses
    MC-dropout to compute the model uncertainty. The 'hERG', 'Nav1.5', and 'Cav1.2' columns contain the
    binary predictions (0 or 1) for each target, representing non-toxic (negative class) or toxic (positive class)
    compounds, respectively.

    Parameters:
        smiles_list: List[str]
            A list containing SMILES strings of chemical compounds.

    Returns:
        Dict[str, any]:
            The function returns the predictions as a dictionary.
    """

    # Compute features
    glv.LOGGER.info("Calculate Features!")
    fingerprints = compute_fingerprint_features(smiles_list)
    descriptors = compute_descriptor_features(smiles_list)
    # Process hERG
    glv.LOGGER.info("Predict hERG toxicity!")
    hERG_fingerprints = fingerprints
    ## Load model
    hERG_predictor = hERGClassifier(1905, 2, glv.GLB_DROPOUT_RATE)
    path = ["CToxPred2", "models", "model_weights", "hERG", "_herg_checkpoint.model"]
    hERG_predictor.load(os.path.join(*path))
    device = torch.device("cpu")
    all_predictions = torch.empty(len(hERG_fingerprints), 2, glv.GLB_FORWARD_ITERATIONS)
    hERG_predictor.train()
    for i in range(glv.GLB_FORWARD_ITERATIONS):
        herg_preds = hERG_predictor(
            torch.from_numpy(hERG_fingerprints).float().to(device)
        )
        all_predictions[:, :, i] = herg_preds.cpu()
    mean_predictions = all_predictions.mean(dim=2)
    hERG_confidence_predictions, hERG_predictions = mean_predictions.max(1)

    # Process Nav1.5
    glv.LOGGER.info("Predict Nav1.5 toxicity!")
    nav15_fingerprints = fingerprints
    nav15_descriptors = descriptors
    ## Load preprocessing pipeline
    path = [
        "CToxPred2",
        "models",
        "decriptors_preprocessing",
        "Nav1.5",
        "nav_descriptors_preprocessing_pipeline.sav",
    ]
    descriptors_transformation_pipeline = joblib.load(os.path.join(*path))
    nav15_descriptors = descriptors_transformation_pipeline.transform(nav15_descriptors)
    nav15_features = np.concatenate((nav15_fingerprints, nav15_descriptors), axis=1)
    ## Load model
    nav15_predictor = Nav15Classifier(2453, 2, glv.GLB_DROPOUT_RATE)
    path = ["CToxPred2", "models", "model_weights", "Nav1.5", "_nav15_checkpoint.model"]
    nav15_predictor.load(os.path.join(*path))
    # nav15_predictions = nav15_predictor(
    #    torch.from_numpy(nav15_features).float().to(device)).argmax(1).cpu()

    all_predictions = torch.empty(len(nav15_features), 2, glv.GLB_FORWARD_ITERATIONS)
    nav15_predictor.train()
    for i in range(glv.GLB_FORWARD_ITERATIONS):
        nav15_preds = nav15_predictor(
            torch.from_numpy(nav15_features).float().to(device)
        )
        all_predictions[:, :, i] = nav15_preds.cpu()
    mean_predictions = all_predictions.mean(dim=2)
    nav15_confidence_predictions, nav15_predictions = mean_predictions.max(1)

    # Process Cav1.2
    glv.LOGGER.info("Predict Cav1.2 toxicity!")
    cav12_fingerprints = fingerprints
    cav12_descriptors = descriptors
    ## Load preprocessing pipeline
    path = [
        "CToxPred2",
        "models",
        "decriptors_preprocessing",
        "Cav1.2",
        "cav_descriptors_preprocessing_pipeline.sav",
    ]
    descriptors_transformation_pipeline = joblib.load(os.path.join(*path))
    cav12_descriptors = descriptors_transformation_pipeline.transform(cav12_descriptors)
    cav12_features = np.concatenate((cav12_fingerprints, cav12_descriptors), axis=1)
    ## Load model
    cav12_predictor = Cav12Classifier(2586, 2, glv.GLB_DROPOUT_RATE)
    path = ["CToxPred2", "models", "model_weights", "Cav1.2", "_cav12_checkpoint.model"]
    cav12_predictor.load(os.path.join(*path))
    # cav12_predictions = cav12_predictor(
    #    torch.from_numpy(cav12_features).float().to(device)).argmax(1).cpu()

    all_predictions = torch.empty(len(cav12_features), 2, glv.GLB_FORWARD_ITERATIONS)
    cav12_predictor.train()
    for i in range(glv.GLB_FORWARD_ITERATIONS):
        cav12_preds = cav12_predictor(
            torch.from_numpy(cav12_features).float().to(device)
        )
        all_predictions[:, :, i] = cav12_preds.cpu()
    mean_predictions = all_predictions.mean(dim=2)
    cav12_confidence_predictions, cav12_predictions = mean_predictions.max(1)

    # Output formating
    hERG_predictions, hERG_confidence_predictions = zip(
        *[
            (prediction.item(), "{:.1%}".format(confidence.item()))
            for prediction, confidence in zip(
                hERG_predictions, hERG_confidence_predictions
            )
        ]
    )

    nav15_predictions, nav15_confidence_predictions = zip(
        *[
            (prediction.item(), "{:.1%}".format(confidence.item()))
            for prediction, confidence in zip(
                nav15_predictions, nav15_confidence_predictions
            )
        ]
    )

    cav12_predictions, cav12_confidence_predictions = zip(
        *[
            (prediction.item(), "{:.1%}".format(confidence.item()))
            for prediction, confidence in zip(
                cav12_predictions, cav12_confidence_predictions
            )
        ]
    )

    return {
        "hERG": hERG_predictions,
        "hERG_confidence": hERG_confidence_predictions,
        "Nav1.5": nav15_predictions,
        "Nav1.5_confidence": nav15_confidence_predictions,
        "Cav1.2": cav12_predictions,
        "Cav1.2_confidence": cav12_confidence_predictions,
    }


def process_smiles(smiles_list):
    # Generate corresponding IDs in same order
    ids = range(1, len(smiles_list) + 1)

    # Create a Pandas DataFrame
    data = {"ID": ids, "SMILES": smiles_list}
    base_frame = pd.DataFrame(data)

    # Generate InChI keys
    base_frame["mol_object"] = base_frame.apply(
        lambda x: Chem.MolFromSmiles(x["SMILES"], sanitize=True), axis=1
    )
    base_frame["InChI"] = base_frame.apply(
        lambda x: Chem.MolToInchiKey(x["mol_object"]), axis=1
    )

    # Compute Physicochemical properties
    # data["MOL"] = data["SMILES"].agg(Chem.MolFromSmiles)
    base_frame["prop"] = base_frame["mol_object"].agg(properties)

    # Split Physicochemical properties
    properties_df = base_frame.apply(split_list_to_columns, axis=1)
    base_frame = pd.concat([base_frame, properties_df], axis=1)
    # round
    columns_float = ["MW", "AlogP", "MPSA"]
    columns_int = ["HBA", "HBD", "ROTB", "AROMS", "ALERTS"]
    precision = 2

    base_frame[columns_float] = base_frame[columns_float].astype(float).round(precision)
    base_frame[columns_int] = base_frame[columns_int].astype(int)

    # Draw 2D structure
    base_frame["_2d_structure"] = base_frame.apply(
        lambda x: resolve_2d_structure(x["SMILES"]), axis=1
    )
    if glv.GLB_Model == "dl-sl":
        pred_results = _generate_predictions_sl(smiles_list)
    else:
        pred_results = _generate_predictions_ssl(smiles_list)

    base_frame["hERG"] = pred_results["hERG"]
    base_frame["hERG_confidence"] = pred_results["hERG_confidence"]
    base_frame["Nav1.5"] = pred_results["Nav1.5"]
    base_frame["Nav1.5_confidence"] = pred_results["Nav1.5_confidence"]
    base_frame["Cav1.2"] = pred_results["Cav1.2"]
    base_frame["Cav1.2_confidence"] = pred_results["Cav1.2_confidence"]

    return base_frame.sort_values(by="ID")
