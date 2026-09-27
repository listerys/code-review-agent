from .models import Tenant


class TenantMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.tenant = Tenant.objects.get(site=request.site)
        return self.get_response(request)
