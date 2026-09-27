from django.urls import path

from . import views

app_name = "shop"

urlpatterns = [
    path("orders/", views.OrderListView.as_view(), name="order_list"),
    path("orders/<int:pk>/", views.order_detail, name="order_detail"),
    path("orders/<int:pk>/refund/", views.refund_order, name="refund_order"),
]
