from decimal import Decimal
from unittest import mock

from django.test import TestCase

from shop.services.refunds import RefundService
from shop.tests.factories import OrderLineFactory


class RefundServiceTests(TestCase):
    def test_calculate_refund_applies_restocking_fee(self):
        line = OrderLineFactory()
        service = RefundService(line.order.tenant)
        amount = service.calculate_refund(line.order, [line])
        self.assertEqual(amount, Decimal("90.00"))

    def test_issue_refund_marks_lines_refunded(self):
        line = OrderLineFactory()
        service = RefundService(line.order.tenant)
        with mock.patch("shop.services.refunds.send_refund_email.delay"):
            service.issue_refund(line.order.pk, [line.pk])
        line.refresh_from_db()
        self.assertEqual(line.refunded_quantity, 2)
