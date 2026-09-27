from django.db import migrations
from django.db.models import Sum


def backfill_refunded_totals(apps, schema_editor):
    Order = apps.get_model("shop", "Order")
    for order in Order.objects.all():
        total = order.refunds.aggregate(total=Sum("amount"))["total"] or 0
        order.refunded_total = total
        order.save(update_fields=["refunded_total"])


class Migration(migrations.Migration):
    dependencies = [("shop", "0001_initial")]

    operations = [migrations.RunPython(backfill_refunded_totals, migrations.RunPython.noop)]
