from normalize import vendor_matches, date_matches, total_matches


def test_vendor_handles_suffixes_and_case():
    assert vendor_matches("Acme Corp", "acme corp")
    assert vendor_matches("Acme Corp", "acme")
    assert not vendor_matches("Acme Corp", "kai")


def test_date_handles_formats():
    assert date_matches("2026-07-07", "07/07/2026")
    assert date_matches("July 7, 2026", "2026-07-07")
    assert not date_matches("2026-07-07", "2026-07-08")


def test_total_handles_currency_and_locale():
    assert total_matches("$1,234.50", 1234.50)
    assert total_matches("1.234,50", 1234.50)  # EU format
    assert total_matches(1000, 1000.99, rel_tol=0.001)  # within 0.1%
    assert not total_matches(100, 200)
