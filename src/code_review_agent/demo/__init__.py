from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

DEMO_ROOT = Path(__file__).parent
BASE_FILES = DEMO_ROOT / "base"
FEATURE_FILES = DEMO_ROOT / "feature"
FEATURE_BRANCH = "feature/partial-refunds"

INJECTED_BUGS = [
    {
        "path": "shop/views.py",
        "category": "security.tenant_isolation",
        "summary": "order_detail no longer filters by request.tenant, so any logged-in user can read another tenant's order",
    },
    {
        "path": "shop/services/refunds.py",
        "category": "state.async_before_commit",
        "summary": "send_refund_email.delay runs inside the atomic block instead of transaction.on_commit",
    },
    {
        "path": "shop/services/refunds.py",
        "category": "bug.money_calculation",
        "summary": "calculate_refund ignores refunded_quantity and order.refundable_amount, so repeat refunds over-refund",
    },
    {
        "path": "shop/views.py",
        "category": "bug.input_validation",
        "summary": "requested quantities are not validated (non-integer raises, negative or excessive quantities are accepted)",
    },
]

NOT_A_BUG = [
    {
        "path": "shop/services/refunds.py",
        "summary": "invalidate_order_total was removed, but the Refund and Order post_save receivers still invalidate the cache",
        "terms": ["invalidate_order_total", "cache"],
    }
]


def git(target: Path, *args: str) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Demo Author",
            "GIT_AUTHOR_EMAIL": "demo@example.com",
            "GIT_COMMITTER_NAME": "Demo Author",
            "GIT_COMMITTER_EMAIL": "demo@example.com",
            "GIT_AUTHOR_DATE": "2026-01-01T12:00:00Z",
            "GIT_COMMITTER_DATE": "2026-01-01T12:00:00Z",
        }
    )
    subprocess.run(["git", "-C", str(target), *args], check=True, capture_output=True, env=environment)


def create_demo_repo(target: Path) -> Path:
    target = Path(target).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BASE_FILES, target, dirs_exist_ok=True)
    git(target, "init", "-q", "-b", "main")
    git(target, "add", "-A")
    git(target, "commit", "-q", "-m", "Initial tinyshop with tenant-scoped refunds")
    git(target, "checkout", "-q", "-b", FEATURE_BRANCH)
    shutil.copytree(FEATURE_FILES, target, dirs_exist_ok=True)
    git(target, "add", "-A")
    git(target, "commit", "-q", "-m", "Support partial refunds by quantity")
    return target
