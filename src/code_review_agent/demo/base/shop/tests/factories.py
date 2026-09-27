from decimal import Decimal

import factory
from django.contrib.sites.models import Site

from shop.models import Order, OrderLine, Tenant


class TenantFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Tenant

    site = factory.LazyFunction(lambda: Site.objects.create(domain="a.example.com", name="a"))
    name = "Tenant A"


class OrderFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Order

    tenant = factory.SubFactory(TenantFactory)
    total = Decimal("100.00")


class OrderLineFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = OrderLine

    order = factory.SubFactory(OrderFactory)
    sku = "SKU-1"
    price = Decimal("50.00")
    quantity = 2
