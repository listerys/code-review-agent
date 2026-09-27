from django.test import TestCase
from django.urls import reverse

from shop.tests.factories import OrderFactory


class OrderViewTests(TestCase):
    def test_order_detail_requires_login(self):
        order = OrderFactory()
        response = self.client.get(reverse("shop:order_detail", args=[order.pk]))
        self.assertEqual(response.status_code, 302)

    def test_order_list_renders(self):
        response = self.client.get("/shop/orders/")
        self.assertIn(response.status_code, (200, 302))
