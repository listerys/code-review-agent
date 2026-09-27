from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST
from django.views.generic import ListView

from .models import Order
from .services.pricing import cached_order_total
from .services.refunds import RefundError, RefundService


@login_required
def order_detail(request, pk):
    order = get_object_or_404(Order.objects.select_related("customer"), pk=pk)
    total = cached_order_total(order)
    return render(request, "shop/order_detail.html", {"order": order, "total": total})


class OrderListView(LoginRequiredMixin, ListView):
    template_name = "shop/order_list.html"
    context_object_name = "orders"

    def get_queryset(self):
        return Order.objects.filter(tenant=self.request.tenant).order_by("-created_at")


@login_required
@require_POST
def refund_order(request, pk):
    service = RefundService(request.tenant)
    line_ids = request.POST.getlist("line_ids")
    quantities = {}
    for line_id in line_ids:
        requested = request.POST.get(f"quantity_{line_id}")
        if requested:
            quantities[int(line_id)] = int(requested)
    try:
        refund = service.issue_refund(pk, line_ids, reason=request.POST.get("reason", ""), quantities=quantities)
    except RefundError as error:
        return JsonResponse({"error": str(error)}, status=400)
    return JsonResponse({"refund_id": refund.pk, "amount": str(refund.amount)})
