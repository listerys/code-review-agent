from celery import shared_task

from .emails import send_refund_issued_email


@shared_task(bind=True, max_retries=3)
def send_refund_email(self, refund_id):
    from .models import Refund

    refund = Refund.objects.select_related("order__customer").get(pk=refund_id)
    send_refund_issued_email(refund)


@shared_task
def send_order_receipt(order_id):
    from .models import Order
    from .emails import send_receipt_email

    send_receipt_email(Order.objects.get(pk=order_id))
