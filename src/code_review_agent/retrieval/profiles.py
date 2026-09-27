from __future__ import annotations

from pydantic import BaseModel, Field

from .models import SearchFilters


DEFAULT_RELATION_WEIGHTS: dict[str, float] = {
    "callers": 1.0,
    "callees": 0.8,
    "tests": 0.8,
    "task_callers": 1.0,
    "receivers": 0.9,
    "model_receivers": 0.9,
    "routes": 0.8,
    "templates": 0.7,
    "rendered_by": 0.7,
    "cache_peers": 0.8,
    "signal_senders": 0.6,
    "subclasses": 0.6,
    "bases": 0.6,
    "links": 0.5,
    "admin": 0.4,
    "model_admin": 0.4,
    "queried_by": 0.3,
    "model_queries": 0.3,
    "imports": 0.2,
}


class RetrievalProfile(BaseModel):
    name: str
    focus: str
    relations: list[str]
    relation_depth: int = 1
    relation_cap: int = 20
    relation_weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_RELATION_WEIGHTS))
    fact_quotas: dict[str, int] = Field(default_factory=dict)
    graph_weight: float = 0.5
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    kind_boosts: dict[str, float] = Field(default_factory=dict)
    tag_boosts: dict[str, float] = Field(default_factory=dict)
    lexical_filters: SearchFilters = Field(default_factory=SearchFilters)
    dense_filters: SearchFilters = Field(default_factory=SearchFilters)
    similar_code: bool = False
    include_memory: bool = False
    budget_tokens: int = 14000


PROFILES: dict[str, RetrievalProfile] = {
    "impact_mapper": RetrievalProfile(
        name="impact_mapper",
        focus="Map the blast radius: callers, callees, routes, templates, emails, admin screens, and tests reachable from the changed symbols.",
        relations=[
            "callers",
            "callees",
            "routes",
            "templates",
            "rendered_by",
            "receivers",
            "model_receivers",
            "task_callers",
            "admin",
            "model_admin",
            "queried_by",
            "tests",
            "links",
        ],
        relation_depth=2,
        fact_quotas={"callers": 3, "routes": 2, "templates": 2, "model_receivers": 2, "task_callers": 2, "tests": 2},
        kind_boosts={"url": 0.3, "template_block": 0.3, "test": 0.1},
        tag_boosts={"email": 0.3, "admin": 0.3, "view": 0.3},
        budget_tokens=16000,
    ),
    "bug_hunter": RetrievalProfile(
        name="bug_hunter",
        focus="Find correctness bugs: callers that depend on changed behavior, sibling implementations the change should mirror, edge cases, and tests.",
        relations=["callers", "callees", "tests", "subclasses", "bases"],
        dense_weight=1.3,
        similar_code=True,
        fact_quotas={"callers": 2, "tests": 2},
        dense_filters=SearchFilters(exclude_changed=True, exclude_kinds=["doc", "config", "test"]),
        kind_boosts={"function": 0.1, "method": 0.1},
        tag_boosts={"money": 0.2},
        budget_tokens=14000,
    ),
    "state_regression": RetrievalProfile(
        name="state_regression",
        focus="Find stateful regressions: signal receivers, Celery tasks, transaction and on_commit boundaries, cache keys, and ordering assumptions.",
        relations=["model_receivers", "receivers", "task_callers", "callees", "cache_peers", "signal_senders", "callers"],
        relation_weights=dict(DEFAULT_RELATION_WEIGHTS, model_receivers=1.0, receivers=1.0, cache_peers=1.0, signal_senders=0.8),
        fact_quotas={"model_receivers": 3, "receivers": 2, "task_callers": 3, "cache_peers": 2},
        tag_boosts={
            "signal_receiver": 0.5,
            "celery_task": 0.5,
            "atomic_block": 0.4,
            "cache_write": 0.4,
            "cache_read": 0.2,
            "task_in_atomic_without_on_commit": 0.6,
            "calls_task:*": 0.3,
        },
        lexical_filters=SearchFilters(exclude_kinds=["doc"]),
        budget_tokens=14000,
    ),
    "security_integrity": RetrievalProfile(
        name="security_integrity",
        focus="Find security and integrity issues: tenant isolation, authorization on views, unscoped querysets, money fields, mass assignment, and injection.",
        relations=["routes", "callers", "model_queries", "queried_by", "admin", "callees"],
        relation_weights=dict(DEFAULT_RELATION_WEIGHTS, routes=1.0, model_queries=0.9, queried_by=0.6, admin=0.6),
        fact_quotas={"routes": 2, "model_queries": 3, "callers": 2},
        lexical_weight=1.3,
        tag_boosts={
            "tenant_scoped": 0.4,
            "unscoped_query": 0.7,
            "no_auth_gate": 0.7,
            "auth_gate": 0.2,
            "money": 0.4,
            "csrf_exempt": 0.5,
            "raw_sql": 0.5,
            "view": 0.3,
            "serializer": 0.3,
            "form": 0.2,
        },
        kind_boosts={"model": 0.2, "url": 0.2},
        budget_tokens=14000,
    ),
    "test_verifier": RetrievalProfile(
        name="test_verifier",
        focus="Verify claims and find test gaps: tests and factories that exercise the changed symbols, and similar past bugs.",
        relations=["tests", "callers"],
        include_memory=True,
        fact_quotas={"tests": 5, "callers": 2},
        kind_boosts={"test": 0.6},
        tag_boosts={"factory": 0.4, "fixture": 0.3, "test_class": 0.2},
        budget_tokens=14000,
    ),
}


def profile_for(name: str) -> RetrievalProfile:
    if name not in PROFILES:
        raise KeyError(f"unknown retrieval profile {name}; choose from {', '.join(PROFILES)}")
    return PROFILES[name]
