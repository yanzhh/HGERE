"""Vocabulary utilities for GSAP-ERE.

The vocabulary is bundled as a package resource at
``hgere/resources/gsap_ere_vocabulary.json``.  It is a JSON array of strings
where ``vocab[token_id] == token_string``.

GSAP-ERE JSONL files downloaded from the BERD platform encode all tokens as
integer IDs.  Use :func:`replace_token_ids` to convert a raw document dict
to human-readable string tokens before passing it to the dataset loaders.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path


def load_gsap_ere_vocabulary() -> list[str]:
    """Load the bundled GSAP-ERE vocabulary.

    Returns
    -------
    list[str]
        Vocabulary list where ``vocab[token_id]`` is the corresponding token
        string.  Length is 20 741.
    """
    resource = files("gsapere") / "resources" / "gsap_ere_vocabulary.json"
    return json.loads(resource.read_text(encoding="utf-8"))


def replace_token_ids(
    doc: dict,
    vocab: list[str],
) -> dict:
    """Replace integer token IDs with string tokens in a GSAP-ERE document.

    Operates on the ``sentences`` and ``doc_tokens`` fields in-place on a
    shallow copy of *doc*.  Fields that are already strings are left unchanged.

    Parameters
    ----------
    doc:
        Raw document dict as loaded from a BERD-platform JSONL file.
    vocab:
        Vocabulary list returned by :func:`load_gsap_ere_vocabulary`.

    Returns
    -------
    dict
        New document dict with integer IDs replaced by string tokens.
    """
    doc = dict(doc)

    if "sentences" in doc:
        sentences = doc["sentences"]
        if sentences and isinstance(sentences[0][0], int):
            doc["sentences"] = [[vocab[i] for i in sent] for sent in sentences]

    if "doc_tokens" in doc:
        tokens = doc["doc_tokens"]
        if tokens and isinstance(tokens[0], int):
            doc["doc_tokens"] = [vocab[i] for i in tokens]

    return doc


def convert_jsonl(src: Path, dst: Path, vocab: list[str]) -> int:
    """Convert a GSAP-ERE JSONL file from token IDs to string tokens.

    Parameters
    ----------
    src:
        Source JSONL file with integer token IDs.
    dst:
        Destination path for the converted file (may equal *src*).
    vocab:
        Vocabulary list returned by :func:`load_gsap_ere_vocabulary`.

    Returns
    -------
    int
        Number of documents converted.
    """
    docs = []
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(replace_token_ids(json.loads(line), vocab))

    tmp = dst.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    tmp.replace(dst)

    return len(docs)
