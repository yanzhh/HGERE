from pathlib import Path
import json

def remove_low_confidence_predictions(data_in:list, confidence_threshold:float=0.001) -> list:
    output_list = []
    for predicted_ner_proba_sentence in data_in:
        sentence_list = []
        for predicted_ner_proba_item in predicted_ner_proba_sentence:
            span_start, span_end, confidence, tag = predicted_ner_proba_item
            if confidence < confidence_threshold:
                continue
            else:
                out_list_temp = [span_start, span_end, tag]
                sentence_list.append(out_list_temp)
        output_list.append(sentence_list)
    return output_list


if __name__ == "__main__":
    input_path = Path("/home/groups/gsap/gsap-ere/models/pruner/default/ent_pred_2025-05-19_test.json")
    output_path = Path("/home/groups/gsap/gsap-ere/models/pruner/default/ent_pred_2025-05-19_test_filtered.json")
    input_file = None
    output_temp = []
    with open(input_path, "r") as f:
        input_file = f.read().split("\n")
    for json_dict in input_file:
        if json_dict != "":    
            parsed_input = json.loads(json_dict)
            pred_ner_proba_section = parsed_input["predicted_ner_proba"]
            output = remove_low_confidence_predictions(pred_ner_proba_section)
            # diff_length = len(pred_ner_proba_section) - len(output)
            # print(diff_length)
            parsed_input["predicted_ner"] = output
            output_temp.append(json.dumps(parsed_input))
    output_str = "\n".join(output_temp)
    with open(output_path, "w+") as f:
        f.write(output_str)
