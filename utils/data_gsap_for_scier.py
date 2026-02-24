from pathlib import Path
import sys
import json


def rename_annotation_layer(path_docs, suffix="_gsap"):
    path_docs = Path(path_docs)
    path_docs_target = path_docs.parent / path_docs.name.replace(
        ".json", "_wo_rels.json"
    )

    with path_docs.open("r") as f:
        with path_docs_target.open("w") as f_t:
            docs = []
            for line in f:
                doc = json.loads(line)
                doc[f"relations{suffix}"] = doc["relations"]
                doc["relations"] = [[] for _ in range(len(doc["relations"]))]
                doc[f"ner{suffix}"] = doc["ner"]
                doc["ner"] = [[] for _ in range(len(doc["ner"]))]
                docs.append(json.dumps(doc))
            f_t.write("\n".join(docs))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path_docs = sys.argv[1]
        rename_annotation_layer(path_docs)

    else:
        print("path to docs needed")
