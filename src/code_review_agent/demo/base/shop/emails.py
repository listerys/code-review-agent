from django.conf import settings
from django.core.mail import send_mail
from django.template.loader import render_to_string


def send_refund_issued_email(refund):
    customer = refund.order.customer
    body = render_to_string("emails/refund_issued.html", {"refund": refund, "order": refund.order})
    send_mail("Your refund", body, settings.DEFAULT_FROM_EMAIL, [customer.email], html_message=body)


def send_receipt_email(order):
    body = render_to_string("emails/receipt.html", {"order": order})
    send_mail("Your receipt", body, settings.DEFAULT_FROM_EMAIL, [order.customer.email], html_message=body)
