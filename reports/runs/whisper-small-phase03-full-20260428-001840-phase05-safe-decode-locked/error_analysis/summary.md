# Phase 04 Error Analysis: whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked

This report turns prediction-level outputs into concrete failure buckets for follow-up training decisions.

## Inputs

- Predictions CSV: `reports/runs/whisper-small-phase03-full-20260428-001840-phase05-safe-decode-locked/predictions.csv`
- Source manifest: `dataset/metadata_test.csv`
- Total samples: `1036`

## Overall Metrics

- Corpus WER: `0.382086`
- Corpus CER: `0.174496`
- Short clip threshold: `< 3.0s`
- Long clip threshold: `>= 10.0s`

## Main Findings

- Code-switching is still a distinct regime in the test set: average WER is 0.369589 on 595 code-switched samples versus 0.417463 on 441 Arabic-only samples.
- Duration still matters: short clips under 3.0s reach corpus WER 0.406465, while clips at or above 10.0s reach 0.378032.
- A small set of repetition loops is driving the worst failures: 1 samples show repeated-token hallucination, including 0 catastrophic loops with WER above 2.0.
- Major omissions are rare but real: 18 samples collapse into much shorter predictions, often on very short or Latin-heavy references.
- Some errors preserve the rough word skeleton but corrupt token forms: 22 samples land in the high-CER / moderate-WER bucket.

## Bucket Summary

### code_switched_reference

- Description: Reference contains Latin-script code-switching.
- Sample count: `595` / `1036` (`57.43%`)
- Average WER: `0.369589`
- Average CER: `0.188712`
- Corpus WER: `0.366600`
- Corpus CER: `0.178332`

### arabic_only_reference

- Description: Reference is Arabic-only with no Latin-script words.
- Sample count: `441` / `1036` (`42.57%`)
- Average WER: `0.417463`
- Average CER: `0.174115`
- Corpus WER: `0.417612`
- Corpus CER: `0.165530`

### short_clip

- Description: Clip duration is shorter than 3.0 seconds.
- Sample count: `318` / `1036` (`30.69%`)
- Average WER: `0.428537`
- Average CER: `0.212853`
- Corpus WER: `0.406465`
- Corpus CER: `0.191824`

### long_clip

- Description: Clip duration is at least 10.0 seconds.
- Sample count: `134` / `1036` (`12.93%`)
- Average WER: `0.364510`
- Average CER: `0.155380`
- Corpus WER: `0.378032`
- Corpus CER: `0.165384`

### high_cer_moderate_wer

- Description: Character corruption is high even when word error is only moderate.
- Sample count: `22` / `1036` (`2.12%`)
- Average WER: `0.502734`
- Average CER: `0.444841`
- Corpus WER: `0.520231`
- Corpus CER: `0.436195`

### major_omission

- Description: Prediction is much shorter than the reference and misses large content.
- Sample count: `18` / `1036` (`1.74%`)
- Average WER: `0.897917`
- Average CER: `0.773333`
- Corpus WER: `0.853933`
- Corpus CER: `0.709534`

### repeated_token_hallucination

- Description: Prediction contains repeated token loops not present in the reference.
- Sample count: `1` / `1036` (`0.10%`)
- Average WER: `1.047619`
- Average CER: `0.760000`
- Corpus WER: `1.047619`
- Corpus CER: `0.760000`

### catastrophic_looping

- Description: A severe repetition loop causes extremely large WER.
- Sample count: `0` / `1036` (`0.00%`)
- Average WER: `nan`
- Average CER: `nan`
- Corpus WER: `nan`
- Corpus CER: `nan`

## Worst Samples

- `sample_00015159` | duration `0.708` | WER `2.000000` | CER `1.750000` | flags `code_switched_reference, short_clip`
  ref: stil
  pred: c'est bien
- `sample_00014180` | duration `0.718` | WER `2.000000` | CER `1.200000` | flags `code_switched_reference, short_clip`
  ref: bravo
  pred: برا vous
- `sample_00008754` | duration `2.924` | WER `1.600000` | CER `0.821429` | flags `arabic_only_reference, short_clip`
  ref: ندوة ندوة يسميوها ندوة قرطبة
  pred: نلوها نلوه يسمدوه انا نلقوا تقول لك لبعدي
- `sample_00013361` | duration `1.420` | WER `1.500000` | CER `1.100000` | flags `code_switched_reference, short_clip`
  ref: soit يهز قهوته ويمشي
  pred: c'est why he has come out ويمشي
- `sample_00009120` | duration `1.420` | WER `1.500000` | CER `0.625000` | flags `arabic_only_reference, short_clip`
  ref: وننساوا المساواة
  pred: وانا تساوا نصولوا
- `sample_00001882` | duration `0.880` | WER `1.500000` | CER `0.454545` | flags `arabic_only_reference, short_clip`
  ref: خمستعشن الف
  pred: في مستعشن نل
- `sample_00013319` | duration `2.380` | WER `1.400000` | CER `0.464286` | flags `arabic_only_reference, short_clip`
  ref: انحب انفيق لروحي انفيق لوقتي
  pred: انحبت في اقل روحي في عقل وقتي
- `sample_00008978` | duration `1.381` | WER `1.333333` | CER `0.947368` | flags `code_switched_reference, short_clip`
  ref: j'adore cette photo
  pred: فهمت نعرش انت نقول
- `sample_00015669` | duration `4.041` | WER `1.250000` | CER `0.971831` | flags `code_switched_reference`
  ref: treize janvier deux mille dix sept بدينا نخدموا فيه فعليا كان ouverture
  pred: فارثور و لي بدينا نخدموا فيه فعلايا كان وقت ال treize janvier deux mille dix huit
- `sample_00011666` | duration `2.167` | WER `1.250000` | CER `0.789474` | flags `code_switched_reference, short_clip`
  ref: mais استانست او ثمة
  pred: et c'était نست اه ثمة ام

## Manual Review Set

See `manual_review_candidates.csv` for a fixed set of high-value examples to inspect by hand.
