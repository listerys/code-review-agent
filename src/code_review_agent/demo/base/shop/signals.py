from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Order, Refund
from .services.pricing import invalidate_order_total
from .tasks import send_order_receipt


@receiver(post_save, sender=Order)
def order_saved(sender, instance, created, **kwargs):
    invalidate_order_total(instance.pk)
    if instance.status == "paid":
        transaction.on_commit(lambda: send_order_receipt.delay(instance.pk))


def refund_saved(sender, instance, created, **kwargs):
    if created:
        invalidate_order_total(instance.order_id)


post_save.connect(refund_saved, sender=Refund, dispatch_uid="shop.refund_saved")
