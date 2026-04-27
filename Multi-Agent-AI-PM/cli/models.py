from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    FUNDAMENTALS = "fundamentals"


class RiskProfile(BaseModel):
    experience: str
    income: str
    net_worth: str
    goal: str
    risk: str
    period: str
