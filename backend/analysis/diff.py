import difflib
import html

def token_diff(a_tokens, b_tokens):
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens)
    changes = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        changes.append({
            "op": tag,  # equal, replace, delete, insert
            "a": a_tokens[i1:i2],
            "b": b_tokens[j1:j2]
        })
    return changes


def html_escape(s): return html.escape(s)


def tokens_to_html(tokens, cls):
    if not tokens: return ""
    return f"<span class='{cls}'>" + html_escape(" ".join(tokens)) + "</span>"


def html_token_diff(a_tokens, b_tokens):
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(tokens_to_html(a_tokens[i1:i2], "pa-eq"))
        elif tag == "replace":
            parts.append(tokens_to_html(a_tokens[i1:i2], "pa-del"))
            parts.append(tokens_to_html(b_tokens[j1:j2], "pa-ins"))
        elif tag == "delete":
            parts.append(tokens_to_html(a_tokens[i1:i2], "pa-del"))
        elif tag == "insert":
            parts.append(tokens_to_html(b_tokens[j1:j2], "pa-ins"))
    return " ".join([p for p in parts if p])


def unified_token_diff(a_tokens, b_tokens, a_label="A", b_label="B"):
    a = " ".join(a_tokens).splitlines(keepends=False)
    b = " ".join(b_tokens).splitlines(keepends=False)
    diff = difflib.unified_diff(a, b, fromfile=a_label, tofile=b_label, lineterm="")
    return "\n".join(diff)