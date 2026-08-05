"""
Simple global conversation buffer (single shared memory, no per-user sessions).
Good enough for a solo/demo deployment.
"""

_history = []  # list of (question, answer) tuples
_MAX_TURNS = 5  # keep last N exchanges to avoid unbounded prompt growth


def add_turn(question: str, answer: str):
    _history.append((question, answer))
    if len(_history) > _MAX_TURNS:
        _history.pop(0)


def get_history_text() -> str:
    if not _history:
        return "No previous conversation."
    lines = []
    for q, a in _history:
        lines.append(f"User: {q}\nBot: {a}")
    return "\n".join(lines)


def clear_history():
    _history.clear()