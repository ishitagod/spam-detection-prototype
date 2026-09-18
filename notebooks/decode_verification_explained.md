# How we turn SMS bytes into text — simple explainer

Goes with `decode_verification.ipynb`. Read this first, then open the notebook and run it.

## The problem

A message doesn't arrive as text. It arrives as a hex string like `"48656c6c6f"`.
We have to figure out (a) which "alphabet" those bytes are written in, and
(b) whether there's some non-text junk glued onto the front we need to remove
first. Get either wrong and you get garbled symbols instead of the real message.

## The steps, in order

**1. Turn hex into bytes.**
`bytes.fromhex("48656c6c6f")` → raw bytes. Nothing clever here, just the starting point.

**2. Check for a header (UDH) and remove it.**
Some messages have a few extra bytes stuck on the front that aren't part of the
message — they're instructions like "this is part 2 of 3" or "this is for app X,
not a person." This is detected by shape, not by trusting a flag: a real header
declares its own length, then a sequence of (id, length, value) pieces that must
exactly fill that declared length. Real text essentially never does that by
accident, so this detection is reliable.

**3. Look up which alphabet the message claims to use (`dcs` number).**
Each message carries a small number, `dcs`, that's supposed to say "I'm GSM-7"
or "I'm ASCII" or "I'm Unicode" etc. There's a table: `dcs` number → decoder.
There are two separate tables, one per data source, because the same `dcs`
number means a different thing depending which system the message came from.
One source stores GSM-7 messages pre-expanded to one byte per character; the
other stores them still bit-packed (see next point). Using the wrong table on
the wrong source produces gibberish.

**4. Decode using that alphabet.**
- **ASCII / Latin-1 / UTF-16** — standard, built into Python. Nothing custom.
- **GSM-7** — the odd one out. It only needs 7 bits per character, so to save
  space, characters get packed edge-to-edge across byte boundaries (8 characters
  squeezed into 7 bytes). Before you can read it, you have to *unpack* those
  bits back into individual 7-bit numbers, then look each number up in a
  character table (a fixed lookup table defined by the telecom standard — not
  something you calculate, just something you look up, like ASCII's own table).

**5. If the `dcs` number is ambiguous or unknown, guess — carefully.**
A few `dcs` numbers aren't clearly documented, or the spec lists more than one
possible meaning. For those, every plausible decoder is tried, and whichever
one produces the most legible-looking result is kept. "Most legible" is
measured with the formula below.

## The two formulas, and why they exist

**`printable_score` — "does this decoded text look real?"**

```
printable_score = (characters that are normal & printable) / (total characters)
```

Real text scores near 1.0. A wrong decode usually produces a lot of
"I don't recognize this character" placeholders, so it scores low. This is
used to pick a winner when guessing between decoders (step 5) and to check the
overall health of a `dcs` group in the summary table.

One limitation: GSM-7's lookup table maps almost every possible byte value to
*some* printable character — so binary data decoded (wrongly) as GSM-7 can
still score close to 1.0, purely by coincidence, not because the decode is
actually correct.

## `dcs=4` on the SS7 source

`dcs=4` doesn't mean "text" at all — per the spec, it means "raw binary data
for some app, not a message a person typed."

- About a third of the time, the binary data still has a header (step 2)
  saying what app it belongs to — stripping that header reveals real,
  legible text underneath. That text is recovered normally.
- The rest of the time, there's no header and genuinely no way to know what
  the binary data means. Those are marked "unknown" rather than guessed at.
