# Transcript Normalization Policy

## Purpose

This policy defines how transcript text is normalized for:

- cleaned dataset manifests
- training labels
- evaluation references
- WER and CER computation

The goal is not to make the text look more formal. The goal is to reduce label noise, remove annotation artifacts, and keep spoken content consistent across training and evaluation.

Normalization is implemented in [dataset/text_normalization.py](/home/coworky/Study/Deeplearning/TUN_STT_MODEL/dataset/text_normalization.py).

Current version: `v1`

## What We Observed In This Dataset

The current `metadata_clean.csv` corpus shows several important patterns:

- About `11,903` rows contain Latin-script text
- About `11,037` rows mix Arabic and Latin in the same utterance
- `213` rows contain language annotation tags like `<\fr>` or `<\en>`
- `97` rows contain invisible BOM/zero-width characters
- `1,236` rows have Arabic-to-Latin script collisions without spaces
- `108` rows have Latin-to-Arabic script collisions without spaces
- `339` rows contain repeated whitespace
- `12,533` rows contain Arabic alef variants `أ/إ/آ`
- `4,143` rows contain `ى`

This tells us the corpus is:

- strongly code-switched
- lightly noisy at the formatting layer
- orthographically inconsistent in common Arabic variants

## Normalization Rules

### Keep

These are preserved because they represent spoken content or useful lexical information:

- Tunisian Derja words
- French and English code-switched words
- apostrophes inside Latin contractions such as `l'axe`, `c'est`, `j'ai`
- hesitation-only utterances, even when their orthography is normalized
- digits as written, for now

### Normalize

These are changed consistently:

1. Unicode normalization
   - Apply `NFKC`

2. Invisible characters
   - Remove BOM and zero-width formatting characters

3. Language annotation tags
   - Remove tags such as `<\fr>`, `<\en>`, `/fr>`, `/en>`
   - These are annotation artifacts, not spoken words

4. Latin casing
   - Lowercase all Latin-script text
   - This reduces sparsity for French and English tokens

5. Arabic orthographic variants
   - `أ -> ا`
   - `إ -> ا`
   - `آ -> ا`
   - `ى -> ي`
   - This improves consistency for training and metric computation

6. Script-boundary spacing
   - Insert a space when Arabic and Latin scripts are glued together
   - Example: `périodeال -> période ال`
   - Example: `بالrythme -> بال rythme`

7. Whitespace
   - Collapse repeated whitespace to one space
   - Trim leading and trailing whitespace

## Deliberate Non-Rules

These are intentionally not done in `v1`:

- no removal of apostrophes from French words
- no conversion of digits to words
- no punctuation restoration
- no deletion of short hesitation-only utterances
- no conversion of `ة` to `ه`
- no conversion of `ئ` or `ؤ`

These may be revisited later if error analysis shows they matter.

## Examples

- `Aphrodite<\fr> كان` -> `aphrodite كان`
- `expérience professionnelle/fr>نلقى` -> `expérience professionnelle نلقي`
- `بالrythme` -> `بال rythme`
- `إسمها أم التمر` -> `اسمها ام التمر`
- `آم` -> `ام`

## Data Contract

- `metadata_all.csv` remains the raw source text
- derived manifests may contain:
  - `text_raw`: original transcript
  - `text`: normalized transcript used by the pipeline
- every downstream training or evaluation step must use the normalized `text` field

## Why This Policy Fits This Project

This project is trying to improve Tunisian Derja ASR, not Arabic spelling correction.

That means:

- spoken content matters more than strict formal orthography
- code-switching must stay visible
- annotation artifacts must not be treated as target tokens
- training and evaluation must share the same normalization logic
