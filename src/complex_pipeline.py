"""
Complex multi-agent pipeline — sequential agent chain with decision gates.
Flow: Understand → Judge → Search Decider → [Conditional Search & Recursion] → Comparator → Reporter.
"""
from src.agents.understand import understand
from src.agents.judge import judge
from src.agents.search_decider import search_decide
from src.agents.comparator import compare
from src.agents.reporter import report
from src.tools import web_search


# Decision gate thresholds (tunable via RL later)
DEFAULT_GATE = {
    "fake_conf": 0.6,
    "common_sense_conflict": 0.6,
    "search_need": 0.7,
}


def run_complex_pipeline(
    claim: str,
    gate_params: dict | None = None,
    max_search_reports: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Run the full multi-agent chain with decision gates.
    If (fake_conf > 0.6 AND conflict > 0.6 AND search_need > 0.7), trigger search.
    """
    gate = {**DEFAULT_GATE, **(gate_params or {})}
    fc = gate["fake_conf"]
    cc = gate["common_sense_conflict"]
    sn = gate["search_need"]

    # Step 0: Understand
    if verbose:
        print("\n" + "=" * 60)
        print("FAKE NEWS VERIFIER - Complex Multi-Agent Pipeline")
        print("=" * 60)
        print(f"Input: {claim}\n")

    u = understand(claim)
    if verbose:
        print(f"[0] Understand: statement = {u['statement'][:80]}...")
        if u["key_facts"]:
            print(f"    key_facts = {u['key_facts']}")

    # Step 1: Judge
    j = judge(u["statement"], u["key_facts"])
    if verbose:
        print(f"    Judge: fake_conf = {j['fake_conf']}, common_sense_conflict = {j['common_sense_conflict']}")

    # Step 2: Search Decider
    sd = search_decide(j["fake_conf"], j["common_sense_conflict"])
    search_need_val = sd["search_need"]
    if verbose:
        print(f"    Search Decider: search_need = {search_need_val}")

    # Decision gate: trigger search?
    search_triggered = (
        j["fake_conf"] > fc
        and j["common_sense_conflict"] > cc
        and search_need_val > sn
    )

    searched_reports: list[str] = []
    searched_scores: list[dict] = []

    if search_triggered:
        if verbose:
            print(f"\n    Gate TRIGGERED (fake>{fc}, conflict>{cc}, need>{sn}) → search")
        reports = web_search(u["statement"], max_results=max_search_reports)
        searched_reports = reports

        # Recursive: apply Understand + Judge to each report
        for r in reports:
            u_r = understand(r)
            j_r = judge(u_r["statement"], u_r["key_facts"])
            searched_scores.append(j_r)
        if verbose:
            for i, (rep, sc) in enumerate(zip(reports, searched_scores)):
                print(f"    Report {i+1}: fake={sc['fake_conf']:.2f} — {rep[:60]}...")
    else:
        if verbose:
            print(f"\n    Gate NOT triggered → skip search")

    # Step 4: Comparator
    comp = compare(
        j["fake_conf"],
        j["common_sense_conflict"],
        searched_scores,
    )

    # Step 5: Reporter
    out = report(
        claim,
        u,
        j,
        search_need_val,
        search_triggered,
        searched_reports,
        searched_scores,
        comp,
    )

    if verbose:
        print(f"\n[Reporter] {out['summary']}")
        print()

    return out
