import os
import sys
import json
import subprocess

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def run(config: Dict[str, Any]) -> None:
    """
    Executes the pruner step in an NLP pipeline.

    Steps:
    1. Builds command-line arguments from the pruner config and runs the pruner script.
    2. Loads the output predictions from the pruner.
    3. Applies a threshold filter to prune predictions.
    4. Saves the pruned predictions to a specified target directory.

    Parameters:
        config: Configuration dictionary containing:
            - pruner: {
                - params: dictionary of parameters passed to `run_pruner.py`,
                - threshold: minimum probability to keep a prediction,
                - target_dir: where to store the pruned output
              }
    """
    params_pruner_dict = config["pruner"]["params"]
    params_pruner = get_as_commandline_params(params_pruner_dict)
    cmd = ["python", "run_pruner.py"] + params_pruner

    subprocess.run(cmd, check=True)

    path_model = Path(params_pruner_dict["output_dir"])
    test_file = Path(params_pruner_dict["test_file"])
    fn_pruned_prediction = f'ent_pred_{test_file.with_suffix(".json")}'
    path_pruned_prediction = path_model / fn_pruned_prediction
    
    fn_pruned_prediction_copy = f'mention_candidates_{test_file.with_suffix(".json")}'
    path_pruned_prediction_copy = path_model / fn_pruned_prediction_copy

    path_target = Path(config["target_dir"])
    path_target.mkdir(parents=True, exist_ok=True)
    fn_target_mention_candidates = path_target / test_file.name
    fn_target_mention_candidates_copy = path_target / f"{test_file.stem}_mention_candidates.jsonl"

    threshold = config["pruner"]["threshold"]
    copy_pruned_version(path_pruned_prediction, fn_target_mention_candidates, threshold)
    copy_pruned_version(path_pruned_prediction, fn_target_mention_candidates_copy, threshold)

    # Further steps like running an extraction model could go here.
    params_hgere_dict = config["hgere"]["params"]
    params_hgere = get_as_commandline_params(params_hgere_dict)
    cmd = ["python", "run_hgnn.py"] + params_hgere
    print(cmd)
    subprocess.run(cmd, check=True)
    print("subprocess finished")
    path_pred = Path(params_hgere_dict["output_dir"])
    path_pred /= params_hgere_dict["test_file"]
    path_pred.rename(fn_target_mention_candidates)
    # move file to output
    # * use extraction model for ere on files
    # * save results somewhere


def copy_pruned_version(
    path_pruned_prediction: Union[str, Path],
    path_target: Union[str, Path],
    threshold: float
) -> None:
    """
    Prunes NER predictions in a JSONL file based on a probability threshold.

    For each line in the input file, this function checks the 'predicted_ner_proba' field,
    filters out predictions below the threshold, and writes a new field 'predicted_ner'
    containing the pruned predictions. Each prediction is a list: [begin, end, label].

    Parameters:
        path_pruned_prediction: Path to the input JSONL file.
        path_target: Path to the output file with pruned predictions.
        threshold: Minimum probability to keep a prediction.

    Raises:
        KeyError: If a document does not contain 'predicted_ner_proba'.
    """
    path_pruned_prediction = Path(path_pruned_prediction)
    path_target = Path(path_target)

    with path_pruned_prediction.open("r", encoding="utf-8") as f_in, \
         path_target.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            doc = json.loads(line)

            if "predicted_ner_proba" not in doc:
                raise KeyError("'predicted_ner_proba' field is missing in document")

            pruned = []
            for sentence in doc["predicted_ner_proba"]:
                sentence_pruned = []
                for begin, end, prob, label in sentence:
                    if prob >= threshold:
                        sentence_pruned.append([begin, end, label])
                if not sentence_pruned:
                    # select best (currently at least one pred is needed
                    first = sorted(sentence, key=lambda x: -x[2])[0]
                    begin, end, prob, label = first
                    ent = [begin, end, label]
                    sentence_pruned.append(ent)
                assert sentence_pruned
                pruned.append(sentence_pruned)

            doc["predicted_ner"] = pruned
            f_out.write(json.dumps(doc) + "\n")

   


def get_as_commandline_params(params: Dict[str, Any]) -> List[str]:
    """
    Converts a dictionary of parameters into a list of command-line arguments.

    Parameters:
        params: A dictionary where each key is a parameter name (without '--') and
                each value is its corresponding value. If the value is None, the flag
                is added without a value.

    Returns:
        A list of strings suitable for passing to subprocess or command-line tools.
        Example: {"input": "file.txt", "verbose": None} → ["--input", "file.txt", "--verbose"]
    """
    param_list = []
    for key, value in params.items():
        param_list.append(f"--{key}")
        if value is not None:
            param_list.append(str(value))
    return param_list
   


if __name__ == "__main__":
    fn_config = sys.argv[1]
    devices = sys.argv[2]
    #fn_config = "pipeline/config_gpu.yaml"
    #devices = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    with open(fn_config) as f:
        config = yaml.safe_load(f)
    run(config)
