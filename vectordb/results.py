from dataclasses import dataclass
import numpy as np

@dataclass
class SearchResult:
    id: str
    score: float
    vector: np.ndarray
    metadata: dict