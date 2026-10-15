import re
import unicodedata


class TextCleaner:
    """
    Cleans and normalizes extracted document text before chunking and embedding.
    """

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # 1. Unicode normalization (NFKC decomposes and recomposes canonically)
        text = unicodedata.normalize("NFKC", text)

        # 2. Remove null bytes and non-printable control characters (except newline, tab)
        text = text.replace("\x00", "")
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 3. Standardize linebreaks (\r\n and \r -> \n)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 4. Collapse trailing whitespace on each line
        lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
        text = "\n".join(lines)

        # 5. Collapse excessive empty lines (3 or more newlines become 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 6. Strip leading and trailing whitespace
        return text.strip()
