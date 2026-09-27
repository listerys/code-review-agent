from django.contrib import admin

from .models import Order, OrderLine, Refund


class OrderLineInline(admin.TabularInline):
    model = OrderLine
    extra = 0


@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ("id", "tenant", "customer", "status", "total", "refunded_total")
    list_filter = ("status", "tenant")
    inlines = [OrderLineInline]


admin.site.register(Refund)
