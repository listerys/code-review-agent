import os
import subprocess

from code_review_agent.evals.golden import load_cases, mine_cases
from code_review_agent.evals.retrieval import run_retrieval_eval, score_case
from code_review_agent.git.repo import GitRepo
from code_review_agent.index.builder import open_store, repo_cache_dir, settings_for_repo


def git(root, *args):
    environment = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=environment)


def write(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def build_history(root):
    root.mkdir()
    write(root, "app/__init__.py", "")
    write(root, "app/pricing.py", "def line_total(price, quantity):\n    return price * quantity\n\n\ndef order_total(lines):\n    return sum(line_total(p, q) for p, q in lines)\n")
    write(root, "app/views.py", "from .pricing import order_total\n\n\ndef checkout(request, lines):\n    total = order_total(lines)\n    return total\n")
    write(root, "app/tests/test_pricing.py", "from app.pricing import order_total\n\n\ndef test_total():\n    assert order_total([(2, 3)]) == 6\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    write(root, "app/pricing.py", "def line_total(price, quantity):\n    return price * quantity\n\n\ndef order_total(lines, discount=0):\n    total = sum(line_total(p, q) for p, q in lines)\n    return max(total - discount, 0)\n")
    write(root, "app/views.py", "from .pricing import order_total\n\n\ndef checkout(request, lines):\n    total = order_total(lines, request.discount)\n    return total\n")
    write(root, "app/tests/test_pricing.py", "from app.pricing import order_total\n\n\ndef test_total():\n    assert order_total([(2, 3)]) == 6\n\n\ndef test_discount_floor():\n    assert order_total([(1, 1)], 5) == 0\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "Fix negative totals when discount exceeds order")


def test_score_case_math():
    ranks, recall, reciprocal = score_case(["a", "b", "c", "d"], ["c", "z"], [2, 3])
    assert ranks == [3, None]
    assert recall == {"@2": 0.0, "@3": 0.5}
    assert abs(reciprocal - 1 / 3) < 1e-9


def test_mine_and_evaluate_retrieval(tmp_path):
    root = tmp_path / "history"
    build_history(root)
    repo = GitRepo(root)
    settings = settings_for_repo(repo, use_llm=False)
    cache_dir = repo_cache_dir(repo, settings)
    cases = mine_cases(repo, cache_dir, limit=5)
    assert len(cases) == 1
    case = cases[0]
    assert case.primary_path == "app/pricing.py"
    assert case.view_overrides["app/tests/test_pricing.py"] is not None
    assert list(case.diff_overrides) == ["app/pricing.py"]
    assert [chunk.symbol for chunk in case.relevant] == ["app.views.checkout"]
    assert case.expected_findings[0].path == "app/pricing.py"
    assert load_cases(cache_dir)[0].id == case.id

    report = run_retrieval_eval(repo, settings, open_store(repo, settings), cases, ["bug_hunter"], ["graph", "lexical", "graph+lexical"], [5, 10], enrich=False)
    by_ablation = {summary.ablation: summary for summary in report.summaries}
    assert set(by_ablation) == {"graph", "lexical", "graph+lexical"}
    assert by_ablation["graph"].recall["@5"] == 1.0
    assert by_ablation["graph+lexical"].mrr > 0
