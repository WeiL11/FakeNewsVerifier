"""
Claim extractor: splits input text into atomic claims.
Uses simple sentence splitting; replace with LLM/NLP for production.
"""


def extract_claims(text):
    """Split text into assertion-like sentences (length > 15, contains assertion keywords)."""
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
