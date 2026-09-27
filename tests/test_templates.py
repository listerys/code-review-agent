from code_review_agent.index.templates import parse_template

TEMPLATE = b"""{% extends "base.html" %}
{% load static %}
{% block title %}Order {{ order.pk }}{% endblock %}
{% block content %}
  {% include "partials/total.html" %}
  {% if order.refunded_total %}<p>{{ order.refunded_total|floatformat:2 }}</p>{% endif %}
  {% for line in order.lines.all %}{{ line.sku }}{% endfor %}
  <a href="{% url 'shop:refund_order' order.pk %}">refund</a>
{% endblock %}
"""


def test_template_structure_and_variables():
    parsed = parse_template("shop/templates/emails/order.html", "emails/order.html", TEMPLATE)
    assert parsed.extends == ["base.html"]
    assert parsed.includes == ["partials/total.html"]
    assert [block.block_name for block in parsed.blocks] == ["title", "content"]
    assert "order.refunded_total" in parsed.variables
    assert "order.lines.all" in parsed.variables
    assert parsed.url_names[0][0] == "shop:refund_order"
    assert parsed.is_email
