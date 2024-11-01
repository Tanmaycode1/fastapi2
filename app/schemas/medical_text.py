from pydantic import BaseModel, Field
from typing import List, Optional

class MedicalTextRequest(BaseModel):
    text: str = Field(..., description="Medical text to be parsed")
    include_cost_estimate: bool = Field(default=False, description="Whether to include processing cost estimate")

class PolicyCriteria(BaseModel):
    criteria: List[str] = Field(default_factory=list)
    no: str = Field(default="")
    crit: str = Field(default="")
    desc: str = Field(default="")

class MedicalTextResponse(BaseModel):
    Description: str = Field(default="")
    Applicable_Drug_List: str = Field(default="")
    Policy_Criteria: PolicyCriteria
    logic: str = Field(default="")
    contraindications: str = Field(default="")
    quantity_level_limit: str = Field(default="")
    estimated_cost: Optional[float] = None