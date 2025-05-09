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
SOMD = {
        "ner": [
        'Abbreviation',
        'AlternativeName',
        'Application',
        'Citation',
        'Developer',
        'Extension',
        'License',
        'OperatingSystem',
        'PlugIn',
        'ProgrammingEnvironment',
        'Release',
        'SoftwareCoreference',
        'URL',
        'Version',
            ],
        "rel_sym": [],
        "rel_non_sym": [
            'Version_of',
             'Developer_of',
              'Citation_of',
               'URL_of',
                'PlugIn_of',
                 'Release_of',
                  'Abbreviation_of',
                   'Specification_of',
                    'Extension_of',
                     'License_of',
                      'AlternativeName_of'
            ]
        }

GSAP = {
    "ner": [
        "Method",
        "MLModel",
        "MLModelGeneric",
        "ModelArchitecture",
        "Dataset",
        "DatasetGeneric",
        "DataSource", # old: Datasource
        "Task",
        "ReferenceLink",
        "URL",
    ],
    "rel_sym": ["coreference", "isComparedTo"],
    #"rel_non_sym_old": [
    #    "isBasedOn",
    #    "citation",
    #    "appliedOn",
    #    "evaluatedOn",
    #    "isPartOf",
    #    "trainedOn",
    #    "isHyponymOf",
    #    "hasInstanceType",
    #    "size",
    #    "url",
    #    "versionOf",
    #],
    "rel_non_sym": [
        'citation', # 3
             'usedFor',
              'architecture', #5
                'evaluatedOn',
                 'isPartOf',
                  'trainedOn',
                   'appliedTo',
                    'isHyponymOf', #10
                      'transformedFrom',
                       'benchmarkFor',
                        'generatedBy',
                         'isBasedOn',
                          'sourcedFrom',#15
                           'processed',
                            'hasInstanceType',
                             'size',
                              'url',
                               'versionOf' # 20
                               ]
}
SCIER = {
    "ner": [
        "Method",
        "Dataset",
        "Task"
        ],
    "rel_sym": [],
    "rel_non_sym": [
        "Part-Of",
        "Synonym-Of",
        "Benchmark-For",
        "Used-For",
        "Evaluated-With",
        "Trained-With",
        "SubTask-Of",
        "SubClass-Of",
        "Compare-With"
        ]
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
    "scier": LabelScheme(**SCIER),
    "somd": LabelScheme(**SOMD),
}
