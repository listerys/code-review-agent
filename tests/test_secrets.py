from code_review_agent.guardrails.secrets import scrub, scrub_text


def test_scrubs_known_token_formats_and_keeps_line_count():
    text = "\n".join(
        [
            'OPENAI_API_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"',
            "aws = 'AKIAABCDEFGHIJKLMNOP'",
            "url = 'postgres://admin:hunter2secret@db.internal:5432/app'",
            "token = ghp_abcdefghijklmnopqrstuvwxyz0123456789ABCD",
        ]
    )
    result = scrub_text(text)
    assert "sk-proj-abcdef" not in result.text
    assert "AKIAABCDEFGHIJKLMNOP" not in result.text
    assert "hunter2secret" not in result.text
    assert "ghp_abcdef" not in result.text
    assert result.text.count("\n") == text.count("\n")
    assert len(result.redactions) >= 4


def test_high_entropy_assignment_is_redacted_but_placeholders_are_not():
    assert "«REDACTED" in scrub('STRIPE_SECRET = "k8Zq2vN9xLm4Pw7tR1sY"')
    assert scrub('PASSWORD = "changeme"') == 'PASSWORD = "changeme"'
    assert scrub("SECRET_KEY = os.environ['SECRET_KEY']") == "SECRET_KEY = os.environ['SECRET_KEY']"


def test_private_key_block_body_is_removed():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\nabcdef\n-----END RSA PRIVATE KEY-----\nnext = 1"
    cleaned = scrub(text)
    assert "MIIEpAIBAAKCAQEA" not in cleaned
    assert cleaned.count("\n") == text.count("\n")
    assert cleaned.endswith("next = 1")
