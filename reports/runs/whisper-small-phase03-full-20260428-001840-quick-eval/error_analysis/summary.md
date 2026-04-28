# Phase 04 Error Analysis: whisper-small-phase03-full-20260428-001840-quick-eval

This report turns prediction-level outputs into concrete failure buckets for follow-up training decisions.

## Inputs

- Predictions CSV: `reports/runs/whisper-small-phase03-full-20260428-001840-quick-eval/predictions.csv`
- Source manifest: `dataset/metadata_test.csv`
- Total samples: `20`

## Overall Metrics

- Corpus WER: `0.306383`
- Corpus CER: `0.124402`
- Short clip threshold: `< 3.0s`
- Long clip threshold: `>= 10.0s`

## Main Findings

- Code-switching is still a distinct regime in the test set: average WER is 0.340224 on 10 code-switched samples versus 0.384707 on 10 Arabic-only samples.
- Duration still matters: short clips under 3.0s reach corpus WER 0.500000, while clips at or above 10.0s reach 0.307692.
- Major omissions are rare but real: 1 samples collapse into much shorter predictions, often on very short or Latin-heavy references.

## Bucket Summary

### code_switched_reference

- Description: Reference contains Latin-script code-switching.
- Sample count: `10` / `20` (`50.00%`)
- Average WER: `0.340224`
- Average CER: `0.199350`
- Corpus WER: `0.280822`
- Corpus CER: `0.115385`

### arabic_only_reference

- Description: Reference is Arabic-only with no Latin-script words.
- Sample count: `10` / `20` (`50.00%`)
- Average WER: `0.384707`
- Average CER: `0.141502`
- Corpus WER: `0.348315`
- Corpus CER: `0.139241`

### short_clip

- Description: Clip duration is shorter than 3.0 seconds.
- Sample count: `7` / `20` (`35.00%`)
- Average WER: `0.565193`
- Average CER: `0.312373`
- Corpus WER: `0.500000`
- Corpus CER: `0.237113`

### long_clip

- Description: Clip duration is at least 10.0 seconds.
- Sample count: `2` / `20` (`10.00%`)
- Average WER: `0.302381`
- Average CER: `0.117672`
- Corpus WER: `0.307692`
- Corpus CER: `0.119760`

### high_cer_moderate_wer

- Description: Character corruption is high even when word error is only moderate.
- Sample count: `0` / `20` (`0.00%`)
- Average WER: `nan`
- Average CER: `nan`
- Corpus WER: `nan`
- Corpus CER: `nan`

### major_omission

- Description: Prediction is much shorter than the reference and misses large content.
- Sample count: `1` / `20` (`5.00%`)
- Average WER: `1.000000`
- Average CER: `1.000000`
- Corpus WER: `1.000000`
- Corpus CER: `1.000000`

### repeated_token_hallucination

- Description: Prediction contains repeated token loops not present in the reference.
- Sample count: `0` / `20` (`0.00%`)
- Average WER: `nan`
- Average CER: `nan`
- Corpus WER: `nan`
- Corpus CER: `nan`

### catastrophic_looping

- Description: A severe repetition loop causes extremely large WER.
- Sample count: `0` / `20` (`0.00%`)
- Average WER: `nan`
- Average CER: `nan`
- Corpus WER: `nan`
- Corpus CER: `nan`

## Worst Samples

- `sample_00015532` | duration `0.997` | WER `1.000000` | CER `1.000000` | flags `code_switched_reference, short_clip, major_omission`
  ref: comment
  pred: alors
- `sample_00011983` | duration `2.389` | WER `0.833333` | CER `0.354839` | flags `arabic_only_reference, short_clip`
  ref: والخدمة متاع بعد غدوة تبدا غدوة
  pred: وانخدم انتعقل خدوة تبدا ودوة
- `sample_00018766` | duration `5.950` | WER `0.692308` | CER `0.316456` | flags `arabic_only_reference`
  ref: نقراوا اكا سويعتين بعد نمشيوا نضربولها ملاوي فالليل عرفت عقاب الليل يضربنا الشر
  pred: نقراو اكا ال سواعتين و بعد نضربونا ملاوي عرفت عقاب الليل نضربنا الشهر
- `sample_00012944` | duration `1.194` | WER `0.666667` | CER `0.111111` | flags `arabic_only_reference, short_clip`
  ref: واحنا انقريوا لواش
  pred: واحنا انقاريوا الواش
- `sample_00012208` | duration `2.125` | WER `0.444444` | CER `0.302326` | flags `code_switched_reference, short_clip`
  ref: من غير ما نشعر خرجت الكاميرا وصورت ال scène
  pred: من غير ما نشعر خرجت كني راهو صورت السان
- `sample_00008857` | duration `8.430` | WER `0.428571` | CER `0.157143` | flags `arabic_only_reference`
  ref: فا ثمة اصلاحات هامة تمت علي مستوي هالنخب هذيا اللي هي اتت بثقافة جديدة
  pred: ف ثمة اصلاحات هامة تمت علي وسط وار هالمخب هذيا اللي هي اكتبي ثقافة جديدة
- `sample_00010457` | duration `2.584` | WER `0.428571` | CER `0.125000` | flags `arabic_only_reference, short_clip`
  ref: يعاود لاخر يطلعه الفوق يعاود يجبده للوطة
  pred: يعاود لاخر يطلعوا الفوق يعاود يجبدوا اللوطة
- `sample_00012044` | duration `14.750` | WER `0.371429` | CER `0.152542` | flags `code_switched_reference, long_clip`
  ref: وقت لي لجابة متاعك تولي انا انحب نكون روحي وقتها تولي حققت ال minimum متاع ال ال self satisfaction انت c'est bon راضي علي المس المسار اللي انت مشيت فيه وراضي عاللي قاعد تعمل فيه
  pred: وقت اللي جاي برثايا تتولي انا انحب نكون روحي وقتها تولي ا حققت minimum متاع ال ال ال self satisfaction انت c'est bon راضي علي ال مث المسال اللي انت مشيت فيه وراضي علي قاعدة تعبت فيه
- `sample_00018047` | duration `4.880` | WER `0.363636` | CER `0.166667` | flags `code_switched_reference`
  ref: الي ما يعملوليش it's hard for me to like find products
  pred: الي ما يعملوا ليش it's hard for me to try to like find products
- `sample_00002710` | duration `1.942` | WER `0.333333` | CER `0.133333` | flags `arabic_only_reference, short_clip`
  ref: ما ينجمش يخدم لمتحان كيما غيره
  pred: ما ينجمش يخدم لمتيحة عن كيما غيره

## Manual Review Set

See `manual_review_candidates.csv` for a fixed set of high-value examples to inspect by hand.
