"""
Claim extractor agent — splits input into 1–3 atomic claims (naive for Phase 1).
"""


def extract_claims(text):
    """
    Very naive splitter — in real version use LLM or better NLP.
    For Phase 1: split on periods and look for assertion-like sentences.
    """
    sentences = [
        s.strip()
        for s in text.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    claims = [
        s
        for s in sentences
        if len(s) > 15
        and any(
            word in s.lower()
            for word in ["is", "was", "has", "said", "reported"]
        )
    ]
    return claims if claims else [text]
