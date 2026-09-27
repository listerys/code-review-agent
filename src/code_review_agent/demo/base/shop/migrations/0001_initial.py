from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(name="Tenant", fields=[("id", models.AutoField(primary_key=True))]),
        migrations.CreateModel(name="Order", fields=[("id", models.AutoField(primary_key=True))]),
        migrations.AddField(model_name="order", name="refunded_total", field=models.DecimalField(max_digits=10, decimal_places=2, default=0)),
    ]
