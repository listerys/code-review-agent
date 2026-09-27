from code_review_agent.git.diff import parse_diff

DIFF = """diff --git a/shop/views.py b/shop/views.py
index 1111111..2222222 100644
--- a/shop/views.py
+++ b/shop/views.py
@@ -10,4 +10,5 @@ def order_detail(request, pk):
     order = get(pk)
-    total = order.total
+    total = cached_total(order)
+    audit(order)
     return total
 
diff --git a/shop/new.py b/shop/new.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/shop/new.py
@@ -0,0 +1,2 @@
+def added():
+    return 1
"""


def test_parse_diff_line_numbers_and_statuses():
    diff = parse_diff(DIFF, "base", "head")
    views, new = diff.files
    assert views.path == "shop/views.py" and views.status == "M"
    assert views.added_line_numbers() == {11, 12}
    assert views.removed_line_numbers() == {11}
    assert views.hunks[0].id == "shop/views.py#h1"
    assert new.status == "A" and new.old_path is None
    assert new.added_line_numbers() == {1, 2}


def test_expected_paths_override_labels():
    expected = [("shop/views.py", "shop/views.py", "M"), (None, "shop/new.py", "A")]
    diff = parse_diff(DIFF, "base", "head", expected)
    assert [file.path for file in diff.files] == ["shop/views.py", "shop/new.py"]
    assert diff.hunk("shop/new.py#h1") is not None
