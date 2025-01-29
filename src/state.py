from typing import Dict, List, TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retries: int