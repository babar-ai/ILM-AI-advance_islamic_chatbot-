
from typing import List, Optional, Dict, Any, Annotated


# ─────────────────────────────────────────────────────────────────────────────
# Custom reducer for chat_history
# ─────────────────────────────────────────────────────────────────────────────
# LangGraph calls this function whenever a node returns an updated value for
# a field that has a reducer annotation.
#
# Signature: reducer(current_value, new_value) → merged_value
#
# Behaviour:
#   - "current" is the list already stored in the checkpoint (all past turns).
#   - "update"  is whatever the node returned for this field (new turns to add).
#   - We simply concatenate them so history grows across turns automatically. 
#   - If "update" is None or empty we leave history unchanged (safe default).
# ─────────────────────────────────────────────────────────────────────────────

def _append_chat_history(current: List[Dict], update: List[Dict]) -> List[Dict]:
    
    if not update:
        return current or []

    return (current or []) + update


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph State  (TypedDict — required for Annotated reducers to work)
# ─────────────────────────────────────────────────────────────────────────────
# WHY TypedDict instead of @dataclass?
#   LangGraph's StateGraph reads field annotations at class-definition time to
#   discover reducers.  @dataclass wraps annotations inside its own machinery,
#   which hides the Annotated metadata from LangGraph.  TypedDict exposes them
#   directly, so reducers are detected and called correctly by the graph engine
#   and the checkpointer.
#
# Field guide:
#   user_query        – raw text exactly as the user typed it (never mutated).
#   standalone_query  – context-resolved rewrite used for embedding/retrieval.
#   query_embedding   – dense vector computed once from standalone_query.
#   web_search_results– Tavily results for this turn (replaced each turn).
#   required_sources  – classifier output: which collections to search.
#   classification_reasoning – classifier's reasoning string (debug/trace).
#   retrieved_documents – Qdrant results keyed by source name (replaced each turn).
#   final_response    – LLM answer for this turn (replaced each turn).
#   error_message     – set if any node raises an exception.
#   filters           – optional metadata filters from classifier.
#   stage_timings_ms  – per-request wall-clock duration for each pipeline stage.
#   chat_history      – PERSISTENT across turns via _append_chat_history reducer.
#                       Each element: {"role": "user"|"assistant", "content": "..."}
# ─────────────────────────────────────────────────────────────────────────────

class LangGraphState(dict):
    """
    TypedDict-compatible state for LangGraph with a persistent chat_history
    field that accumulates across turns via the _append_chat_history reducer.
    """
    # Ephemeral fields — reset / overwritten on every turn
    user_query: str
    standalone_query: str
    query_embedding: Optional[List[float]]          # Embed once, reuse everywhere
    web_search_results: List[Dict]                  # Tavily results this turn
    required_sources: List[str]                     # ["quran", "hadith", ...]
    classification_reasoning: str
    retrieved_documents: Dict[str, List[Dict[str, Any]]]
    final_response: str
    error_message: Optional[str]
    filters: Optional[Dict[str, Any]]
    stage_timings_ms: Dict[str, float]

    # Persistent field — grows across turns (checkpointer restores + reducer appends)
    chat_history: Annotated[List[Dict], _append_chat_history]

