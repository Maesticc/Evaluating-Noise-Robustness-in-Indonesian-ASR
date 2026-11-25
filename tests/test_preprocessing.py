from src.preprocessing import normalize_text, number_to_words


def test_number_to_words():
    assert number_to_words(0) == "nol"
    assert number_to_words(11) == "sebelas"
    assert number_to_words(15) == "lima belas"
    assert number_to_words(100) == "seratus"
    assert number_to_words(1000) == "seribu"
    assert number_to_words(2018) == "dua ribu delapan belas"


def test_normalize_text_policy():
    assert normalize_text("Asian Games 2018!") == "asian games dua ribu delapan belas"
    assert normalize_text("Naik 12,5%") == "naik dua belas koma lima persen"
    assert normalize_text("Rp 1.500.000") == "rp satu juta lima ratus ribu"
    assert normalize_text("Anak-anak  bermain.") == "anak anak bermain"
    assert normalize_text("Jum'at café") == "jumat cafe"


def test_normalize_text_is_deterministic_and_handles_missing():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    text = "Harga naik 12,5% dari Rp 1.500.000!"
    assert normalize_text(text) == normalize_text(normalize_text(text))
