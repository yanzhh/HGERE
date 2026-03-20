# Dataset Download

Datasets are downloaded using the `download-dataset` CLI command.

```bash
# List all datasets
uv run download-dataset --list

# Download a full dataset (to datasets/<name>/ relative to cwd)
uv run download-dataset gsap-ere
uv run download-dataset scier
uv run download-dataset scierc
uv run download-dataset scinlp

# Download specific splits only
uv run download-dataset gsap-ere --files train.jsonl dev.jsonl

# Custom output directory
uv run download-dataset gsap-ere --outdir /path/to/output

# Download all datasets at once
uv run download-dataset --all

# Skip MD5 verification
uv run download-dataset gsap-ere --skip-verify
```

The command is implemented in
`src/gsapere/commands/download_dataset.py`.  Each dataset is registered as a
`DatasetSpec` entry in the `DATASETS` dict — adding a new dataset means
filling in a name, description, and a `files` mapping of filename → URL.

---

## Datasets

### GSAP-ERE

Entity and relation extraction dataset for scientific text, developed at GESIS.

| Split | File | Size |
|---|---|---|
| Train | `train.jsonl` | 13.7 MB |
| Dev | `dev.jsonl` | 1.9 MB |
| Test | `test.jsonl` | 1.7 MB |

**Source:** <https://berd-platform.de/records/c4c1d-s0587>
**DOI:** <https://doi.org/10.60914/c4c1d-s0587>

Files are downloaded from the BERD Research Data Repository via InvenioRDM.
MD5 checksums are verified after download.

**Token vocabulary:** The BERD-platform files encode all tokens as integer IDs.
The downloader automatically replaces them with string tokens using the bundled
vocabulary (`src/hgere/resources/gsap_ere_vocabulary.json`, 20 741 tokens).
The vocabulary is sourced from
<https://raw.githubusercontent.com/ottowg/gsap-ere/refs/heads/main/vocabulary.json>.
The bundled vocabulary lives at `src/gsapere/resources/gsap_ere_vocabulary.json`.

---

### SciER

Scientific entity and relation extraction dataset covering Dataset, Method, and
Task entity types across scientific abstracts.

| Split | File |
|---|---|
| Train | `train.jsonl` |
| Dev | `dev.jsonl` |
| Test | `test.jsonl` |
| Test OOD | `test_ood.jsonl` |

**Source:** <https://github.com/edzq/SciER> — `SciER/PLM/` folder (PLM-ready
JSONL format with DyGIE-style `sentences`, `ner`, `relations` fields)

Files are downloaded directly from GitHub raw content.  No checksums are
published by the authors.

---

### SciERC

Scientific information extraction benchmark with 500 annotated abstracts from
12 AI conferences.  Covers 6 entity types and 7 relation types.

| Split | File |
|---|---|
| Train | `train.json` |
| Dev | `dev.json` |
| Test | `test.json` |

**Primary source:** Tsinghua Cloud share
<https://cloud.tsinghua.edu.cn/d/7dafc9a3d84d4151a755/>
**Alternative source:** Google Drive
<https://drive.google.com/drive/folders/1_u6pIe7Dw3Lqy4mF2m1UFqmKmGeM40zS>
**Referenced from:** <https://github.com/thunlp/PL-Marker> (PL-Marker README,
NER data section)

The downloader uses the Tsinghua Cloud source via the Seafile `?dl=1` redirect
endpoint.  No checksums are published by the authors.

If the Tsinghua link stops working, download manually from the Google Drive
alternative and place the three JSON files into `datasets/scierc/`.

---

### SciNLP

Scientific NLP dataset for entity and relation extraction.

| Split | File |
|---|---|
| Train | `train.json` |
| Dev | `dev.json` |
| Test | `test.json` |

**Source:** <https://github.com/AKADDC/SciNLP> — `Dataset/` folder

Files are downloaded directly from GitHub raw content.  No checksums are
published by the authors.

---

## Expected directory layout after download

```
datasets/
  gsap-ere/
    train.jsonl
    dev.jsonl
    test.jsonl
  scier/
    train.jsonl
    dev.jsonl
    test.jsonl
    test_ood.jsonl
  scierc/
    train.json
    dev.json
    test.json
  scinlp/
    train.json
    dev.json
    test.json
```

---

## Adding a new dataset

1. Add a `DatasetSpec` entry to `DATASETS` in
   `src/gsapere/commands/download_dataset.py`.
2. Set `files=None` to mark it as not yet implemented (shows a clear error
   with a `source_hint` pointing to the manual download location).
3. Add a `source_hint` string with the canonical source URL(s) — this is
   printed whenever a download fails so users know where to get the data
   manually.
4. Add a section to this file documenting the source.
