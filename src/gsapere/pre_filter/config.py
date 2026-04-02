"""Pre-filter configuration models.

Defines the discriminated-union type ``PreFilterParams`` used to configure
candidate-span filtering applied before HGERE sees any spans.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

# ---------------------------------------------------------------------------
# Pre-filtering parameter variants
# ---------------------------------------------------------------------------


class ThresholdPreFilterParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["threshold"]
    value: float = Field(ge=0.0, le=1.0)


class TopKPreFilterParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["topk"]
    ratio: float = Field(ge=0.0, le=1.0)
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def check_min_max(self) -> "TopKPreFilterParams":
        if self.max < self.min:
            raise ValueError("max must be >= min")
        return self


PreFilterParams = Annotated[
    Union[ThresholdPreFilterParams, TopKPreFilterParams],
    Field(discriminator="method"),
]
