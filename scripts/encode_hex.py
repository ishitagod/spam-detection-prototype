text = "WIN FREE PRIZE WOOHOO!!!https://dajhsjaksjka.skajksja"

raw = text.encode("utf-16-be")
hex_content = " ".join(f"{b:02x}" for b in raw)

print(hex_content)
