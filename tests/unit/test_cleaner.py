import pytest
from app.ingestion.cleaner import TextCleaner


def test_text_cleaner_null_bytes_and_control_chars():
    dirty = "Hello\x00 World!\x07 This has \x1b control codes."
    clean = TextCleaner.clean(dirty)
    assert "\x00" not in clean
    assert "\x07" not in clean
    assert clean == "Hello World! This has  control codes."


def test_text_cleaner_linebreaks_normalization():
    dirty = "Line 1\r\nLine 2\rLine 3\nLine 4"
    clean = TextCleaner.clean(dirty)
    assert clean == "Line 1\nLine 2\nLine 3\nLine 4"


def test_text_cleaner_excessive_empty_lines():
    dirty = "Paragraph 1\n\n\n\n\nParagraph 2"
    clean = TextCleaner.clean(dirty)
    assert clean == "Paragraph 1\n\nParagraph 2"


def test_text_cleaner_trailing_whitespace():
    dirty = "Line with trailing spaces    \nAnother line\t\t\nFinal line"
    clean = TextCleaner.clean(dirty)
    assert clean == "Line with trailing spaces\nAnother line\nFinal line"


def test_text_cleaner_empty():
    assert TextCleaner.clean("") == ""
    assert TextCleaner.clean("    \n\n  ") == ""
