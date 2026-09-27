from decimal import ROUND_HALF_UP, Decimal

from django.conf import settings
from django.db import transaction

from ..models import Order, Refund
from ..tasks import send_refund_email


class RefundError(Exception):
    pass


class RefundService:
    def __init__(self, tenant):
        self.tenant = tenant

    def restocking_fee(self, amount):
        percent = Decimal(settings.REFUND_RESTOCKING_FEE_PERCENT) / Decimal("100")
        return (amount * percent).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def calculate_refund(self, order, lines, quantities=None):
        gross = Decimal("0.00")
        for line in lines:
            quantity = line.quantity
            if quantities and line.pk in quantities:
                quantity = quantities[line.pk]
            gross += line.price * quantity
        return gross - self.restocking_fee(gross)

    @transaction.atomic
    def issue_refund(self, order_id, line_ids, reason="", quantities=None):
        order = Order.objects.select_for_update().get(pk=order_id, tenant=self.tenant)
        lines = list(order.lines.filter(pk__in=line_ids))
        if not lines:
            raise RefundError("no refundable lines")
        amount = self.calculate_refund(order, lines, quantities)
        refund = Refund.objects.create(order=order, amount=amount, reason=reason)
        for line in lines:
            refunded_now = line.quantity
            if quantities and line.pk in quantities:
                refunded_now = quantities[line.pk]
            line.refunded_quantity = line.refunded_quantity + refunded_now
            line.save(update_fields=["refunded_quantity"])
        order.refunded_total += amount
        order.status = "refunded" if order.refundable_amount() <= 0 else order.status
        order.save(update_fields=["refunded_total", "status"])
        send_refund_email.delay(refund.pk)
        return refund
