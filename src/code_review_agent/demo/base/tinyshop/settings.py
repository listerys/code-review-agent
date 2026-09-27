import os

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only")
DEBUG = False
ALLOWED_HOSTS = ["*"]
SITE_ID = 1

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.sites",
    "shop",
]

MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.sites.middleware.CurrentSiteMiddleware",
    "shop.middleware.TenantMiddleware",
]

ROOT_URLCONF = "tinyshop.urls"

CACHES = {"default": {"BACKEND": "django.core.cache.backends.redis.RedisCache", "LOCATION": "redis://localhost:6379/1"}}

CELERY_BROKER_URL = "redis://localhost:6379/0"
CELERY_TASK_ALWAYS_EAGER = False

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
DEFAULT_FROM_EMAIL = "billing@tinyshop.example.com"
REFUND_RESTOCKING_FEE_PERCENT = 10
