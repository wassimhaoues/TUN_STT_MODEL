# Phase 04 Error Analysis: whisper-small-phase05-postrepair-20260430-124259-locked-test-safe-decode

This report turns prediction-level outputs into concrete failure buckets for follow-up training decisions.

## Inputs

- Predictions CSV: `reports/runs/whisper-small-phase05-postrepair-20260430-124259-locked-test-safe-decode/predictions.csv`
- Source manifest: `dataset/metadata_test.csv`
- Total samples: `1036`

## Overall Metrics

- Corpus WER: `0.375634`
- Corpus CER: `0.172456`
- Short clip threshold: `< 3.0s`
- Long clip threshold: `>= 10.0s`

## Main Findings

- Code-switching is still a distinct regime in the test set: average WER is 0.348044 on 595 code-switched samples versus 0.433658 on 441 Arabic-only samples.
- Duration still matters: short clips under 3.0s reach corpus WER 0.389008, while clips at or above 10.0s reach 0.387573.
- A small set of repetition loops is driving the worst failures: 1 samples show repeated-token hallucination, including 0 catastrophic loops with WER above 2.0.
- Major omissions are rare but real: 18 samples collapse into much shorter predictions, often on very short or Latin-heavy references.
- Some errors preserve the rough word skeleton but corrupt token forms: 10 samples land in the high-CER / moderate-WER bucket.

## CER Review Bands

- Good: `637`
- Acceptable but inspect sometimes: `172`
- Needs review: `104`
- High priority review: `100`
- Critical: `23`

## Bucket Summary

### code_switched_reference

- Description: Reference contains Latin-script code-switching.
- Sample count: `595` / `1036` (`57.43%`)
- Average WER: `0.348044`
- Average CER: `0.172041`
- Corpus WER: `0.358031`
- Corpus CER: `0.174299`

### arabic_only_reference

- Description: Reference is Arabic-only with no Latin-script words.
- Sample count: `441` / `1036` (`42.57%`)
- Average WER: `0.433658`
- Average CER: `0.188212`
- Corpus WER: `0.417753`
- Corpus CER: `0.167950`

### short_clip

- Description: Clip duration is shorter than 3.0 seconds.
- Sample count: `313` / `1036` (`30.21%`)
- Average WER: `0.423558`
- Average CER: `0.211221`
- Corpus WER: `0.389008`
- Corpus CER: `0.179625`

### long_clip

- Description: Clip duration is at least 10.0 seconds.
- Sample count: `135` / `1036` (`13.03%`)
- Average WER: `0.382158`
- Average CER: `0.173367`
- Corpus WER: `0.387573`
- Corpus CER: `0.177809`

### high_cer_moderate_wer

- Description: Character corruption is high even when word error is only moderate.
- Sample count: `10` / `1036` (`0.97%`)
- Average WER: `0.508254`
- Average CER: `0.475630`
- Corpus WER: `0.552239`
- Corpus CER: `0.470939`

### major_omission

- Description: Prediction is much shorter than the reference and misses large content.
- Sample count: `18` / `1036` (`1.74%`)
- Average WER: `0.948148`
- Average CER: `0.666999`
- Corpus WER: `0.932203`
- Corpus CER: `0.687273`

### repeated_token_hallucination

- Description: Prediction contains repeated token loops not present in the reference.
- Sample count: `1` / `1036` (`0.10%`)
- Average WER: `1.428571`
- Average CER: `0.848000`
- Corpus WER: `1.428571`
- Corpus CER: `0.848000`

### catastrophic_looping

- Description: A severe repetition loop causes extremely large WER.
- Sample count: `0` / `1036` (`0.00%`)
- Average WER: `nan`
- Average CER: `nan`
- Corpus WER: `nan`
- Corpus CER: `nan`

## Worst Samples

- `sample_00005573` | duration `0.964` | WER `2.000000` | CER `1.600000` | flags `arabic_only_reference, short_clip`
  ref: مرحبا
  pred: voilà باهي
- `sample_00001891` | duration `0.840` | WER `2.000000` | CER `1.000000` | flags `arabic_only_reference, short_clip`
  ref: عمري
  pred: عام ايه
- `sample_00009194` | duration `1.582` | WER `2.000000` | CER `0.714286` | flags `arabic_only_reference, short_clip`
  ref: نتعداوا
  pred: قاعدة اوه
- `sample_00009794` | duration `1.223` | WER `2.000000` | CER `0.500000` | flags `arabic_only_reference, short_clip`
  ref: ام
  pred: ا م
- `sample_00008978` | duration `1.381` | WER `1.666667` | CER `1.157895` | flags `code_switched_reference, short_clip`
  ref: j'adore cette photo
  pred: فهمت نعرف شوية في القدام
- `sample_00001882` | duration `0.880` | WER `1.500000` | CER `0.727273` | flags `arabic_only_reference, short_clip`
  ref: خمستعشن الف
  pred: وما نستعش نلدل
- `sample_00005861` | duration `9.201` | WER `1.428571` | CER `0.848000` | flags `code_switched_reference, repeated_token_hallucination`
  ref: et les rapports متاعنا en tout cas تكلموا علي رواحهم ومن غاديكا عاونونا بش عملنا لل l'édition الثانية امبعد l'édition الثانية
  pred: et et et et il est à temps en temps quand même on a l'air خمم من غاديكا عامة ما نعرفش عال l'amazonie dix cent cinquante et un quinze et un cent cinque
- `sample_00008754` | duration `2.924` | WER `1.400000` | CER `0.607143` | flags `arabic_only_reference, short_clip`
  ref: ندوة ندوة يسميوها ندوة قرطبة
  pred: نلوة نلوها يسميوه انابو تقول لكم ا
- `sample_00016523` | duration `1.671` | WER `1.333333` | CER `0.285714` | flags `arabic_only_reference, short_clip`
  ref: فرحانين بمقرهم الجديد
  pred: فرحة نيب مقرهم جديد
- `sample_00019089` | duration `12.000` | WER `1.272727` | CER `0.957983` | flags `code_switched_reference, long_clip`
  ref: اللي انا نقول صعيبة برشا خاطرني تعديت période في الكراهب عملت برشا تجارب صغيرة ماللي انا نقري برشا تجارب اما وقتها يمكن
  pred: الاقرب برشا تجاربة اما وقتها كانت يمكن ال période الي انا نقول صعيبة براشا période هاذيكا صحيبة موش خاطرني تعديت حقوقت نخممت عملت يعني التجارب اخري صغير ايه طرش ماللي انا

## Manual Review Set

See `manual_review_candidates.csv` for a fixed set of high-value examples to inspect by hand.
