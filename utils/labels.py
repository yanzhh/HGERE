from typing import List
from dataclasses import dataclass

ACE04 = {
    "ner": ["WEA", "LOC", "VEH", "GPE", "ORG", "PER"],
    "rel_sym": ["PER-SOC"],
    "rel_non_sym": ["OTHER-AFF", "ART", "GPE-AFF", "EMP-ORG", "PHYS"],
}

ACE05 = {
    "ner": ["FAC", "WEA", "LOC", "VEH", "GPE", "ORG", "PER"],
    "rel_sym": ["PER-SOC"],
    "rel_non_sym": ["ART", "ORG-AFF", "GEN-AFF", "PHYS", "PART-WHOLE"],
}
GSAP = {
    "ner": [
        "Method",
        "MLModel",
        "MLModelGeneric",
        "ModelArchitecture",
        "Dataset",
        "DatasetGeneric",
        "Datasource",
        "Task",
        "ReferenceLink",
        "URL",
    ],
    "rel_sym": ["coreference", "isComparedTo"],
    "rel_non_sym": [
        "isBasedOn",
        "citation",
        "appliedOn",
        "evaluatedOn",
        "isPartOf",
        "trainedOn",
        "isHyponymOf",
        "hasInstanceType",
        "size",
        "url",
        "versionOf",
    ],
}
SCIERC = {
    "ner": [
        "Method",
        "OtherScientificTerm",
        "Task",
        "Generic",
        "Material",
        "Metric",
    ],
    "rel_sym": ["CONJUNCTION", "COMPARE"],
    "rel_non_sym": [
        "PART-OF",
        "USED-FOR",
        "FEATURE-OF",
        "EVALUATE-FOR",
        "HYPONYM-OF",
    ],
}


@dataclass
class RelationLabelScheme:
    _sym: List[str]
    _non_sym: List[str]

    @property
    def all(self) -> List[str]:
        return ["NIL"] + self._sym + self._non_sym

    def symmetric(self, only_nil=False) -> List[str]:
        return ["NIL"] if only_nil else ["NIL"] + self._sym


class LabelScheme:
    def __init__(self, ner, rel_sym, rel_non_sym):
        self.ner = ["NIL"] + ner
        self.rel = RelationLabelScheme(rel_sym, rel_non_sym)


LABELS = {
    "ace04": LabelScheme(**ACE04),
    "ace05": LabelScheme(**ACE05),
    "gsap": LabelScheme(**GSAP),
    "scierc": LabelScheme(**SCIERC),
}
