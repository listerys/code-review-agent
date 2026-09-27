# Tinyshop architecture

## Tenancy

Every request resolves a `Tenant` from the current `Site` in `TenantMiddleware`.
Every queryset that touches orders, customers, or refunds must filter by `tenant`.

## Refunds

Refunds are issued by `RefundService.issue_refund`, which locks the order row with
`select_for_update`, records a `Refund`, and enqueues the customer email only after
the transaction commits (`transaction.on_commit`). Order totals are cached under
`order_total:<id>` and must be invalidated whenever a refund changes the order.

## Emails

Customer-facing emails are rendered from `templates/emails/` and sent by Celery tasks.
