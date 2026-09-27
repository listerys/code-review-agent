from code_review_agent.index.python_parser import parse_python

SOURCE = b'''
from django.db import models, transaction
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.cache import cache
from .tasks import send_email
from . import helpers as helper_module


def key_for(order_id):
    return f"order:{order_id}"


class Order(models.Model):
    tenant = models.ForeignKey("shop.Tenant", on_delete=models.CASCADE)
    total = models.DecimalField(max_digits=9, decimal_places=2)

    @transaction.atomic
    def close(self):
        send_email.delay(self.pk)
        transaction.on_commit(lambda: send_email.delay(self.pk))
        key = key_for(self.pk)
        cache.delete(key)
        return Order.objects.filter(tenant=self.tenant).exclude(pk=1).select_for_update().first()


@receiver(post_save, sender=Order)
def order_saved(sender, instance, **kwargs):
    helper_module = instance
    helper_module.touch()


def uses_import():
    helper_module.touch()
'''


def symbols_by_name(parsed):
    return {symbol.qualname: symbol for symbol in parsed.symbols}


def test_symbols_models_and_tags():
    parsed = parse_python("shop/models.py", SOURCE, "shop.models", False, False)
    symbols = symbols_by_name(parsed)
    order = symbols["shop.models.Order"]
    assert order.kind == "model"
    assert [field.name for field in order.fields] == ["tenant", "total"]
    assert order.fields[0].related == "shop.Tenant"
    close = symbols["shop.models.Order.close"]
    assert {"atomic_block", "enqueues_task", "cache_write", "queries_db"} <= close.tags
    assert symbols["shop.models.order_saved"].tags >= {"signal_receiver"}
    assert symbols["shop.models.key_for"].returns_pattern == "order:{}"


def test_task_calls_distinguish_on_commit():
    parsed = parse_python("shop/models.py", SOURCE, "shop.models", False, False)
    task_calls = [ref for ref in parsed.refs if ref.kind == "task_call"]
    flags = sorted((ref.extra["in_atomic"], ref.extra["in_on_commit"]) for ref in task_calls)
    assert flags == [(True, False), (True, True)]


def test_query_chain_collects_methods_and_kwargs():
    parsed = parse_python("shop/models.py", SOURCE, "shop.models", False, False)
    query = [ref for ref in parsed.refs if ref.kind == "query"][0]
    assert query.target == "Order"
    assert query.extra["methods"] == ["filter", "exclude", "select_for_update", "first"]
    assert query.extra["kwargs"] == ["pk", "tenant"]


def test_cache_key_follows_local_variable_to_helper():
    parsed = parse_python("shop/models.py", SOURCE, "shop.models", False, False)
    cache_ref = [ref for ref in parsed.refs if ref.kind == "cache_op"][0]
    assert cache_ref.target == "<call:key_for>"
    assert cache_ref.extra["op"] == "delete"


def test_receiver_and_local_shadowing():
    parsed = parse_python("shop/models.py", SOURCE, "shop.models", False, False)
    receiver = [ref for ref in parsed.refs if ref.kind == "receiver"][0]
    assert receiver.target == "Order" and receiver.extra["signals"] == ["post_save"]
    shadowed = [ref for ref in parsed.refs if ref.kind == "call" and ref.target == "helper_module.touch"]
    local_flags = sorted(bool(ref.extra.get("local")) for ref in shadowed)
    assert local_flags == [False, True]


def test_imports_resolve_relative_modules():
    parsed = parse_python("shop/models.py", SOURCE, "shop.models", False, False)
    targets = {entry.alias: entry.target for entry in parsed.imports}
    assert targets["send_email"] == "shop.tasks.send_email"
    assert targets["helper_module"] == "shop.helpers"


def test_url_patterns():
    source = b'''
from django.urls import include, path, re_path
from . import views

app_name = "shop"
urlpatterns = [
    path("orders/<int:pk>/", views.order_detail, name="order_detail"),
    path("list/", views.OrderList.as_view(), name="order_list"),
    re_path(r"^legacy/$", login_required(views.legacy)),
    path("api/", include("shop.api.urls", namespace="api")),
]
'''
    parsed = parse_python("shop/urls.py", source, "shop.urls", False, False)
    patterns = parsed.url_patterns
    assert [pattern.view_ref for pattern in patterns[:3]] == ["views.order_detail", "views.OrderList", "views.legacy"]
    assert patterns[2].wrappers == ["login_required"] and patterns[2].is_regex
    assert patterns[3].include_ref == "shop.api.urls" and patterns[3].namespace == "api"
    assert parsed.app_name == "shop"


def test_migration_keeps_only_data_functions():
    source = b'''
from django.db import migrations


def forwards(apps, schema_editor):
    Order = apps.get_model("shop", "Order")


class Migration(migrations.Migration):
    operations = [
        migrations.AddField(model_name="order", name="total", field=None),
        migrations.RunPython(forwards, migrations.RunPython.noop),
    ]
'''
    parsed = parse_python("shop/migrations/0003_x.py", source, "shop.migrations.0003_x", False, True)
    operations = [operation["op"] for operation in parsed.migration["operations"]]
    assert operations == ["AddField", "RunPython"]
    assert "forwards" in parsed.data_migration_functions
