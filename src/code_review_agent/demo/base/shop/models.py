from decimal import Decimal

from django.conf import settings
from django.contrib.sites.models import Site
from django.db import models


class Tenant(models.Model):
    site = models.OneToOneField(Site, on_delete=models.CASCADE)
    name = models.CharField(max_length=120)


class Customer(models.Model):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="customers")
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    email = models.EmailField()


class Order(models.Model):
    STATUS_CHOICES = [("open", "Open"), ("paid", "Paid"), ("refunded", "Refunded")]

    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="orders")
    customer = models.ForeignKey(Customer, on_delete=models.PROTECT, related_name="orders")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="open")
    total = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    refunded_total = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    created_at = models.DateTimeField(auto_now_add=True)

    def refundable_amount(self):
        return self.total - self.refunded_total


class OrderLine(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="lines")
    sku = models.CharField(max_length=40)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    quantity = models.PositiveIntegerField(default=1)
    refunded_quantity = models.PositiveIntegerField(default=0)

    def line_total(self):
        return self.price * self.quantity


class Refund(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="refunds")
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    reason = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
