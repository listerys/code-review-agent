from decimal import Decimal

from django.core.cache import cache


def order_total_cache_key(order_id):
    return f"order_total:{order_id}"


def cached_order_total(order):
    key = order_total_cache_key(order.pk)
    total = cache.get(key)
    if total is None:
        total = sum((line.line_total() for line in order.lines.all()), Decimal("0.00"))
        cache.set(key, total, timeout=300)
    return total


def invalidate_order_total(order_id):
    cache.delete(order_total_cache_key(order_id))
