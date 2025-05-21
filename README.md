# How to Use WavBench? Examples of assessment MiniCPM. Tmp Code for Reviewing. All Code will open source soon.

A sample test of the quality of responses to a speech dialogue model is shown here. The test is on the  MiniCPM model, and you can use the dataset we have already collected, or you can collect  MiniCPM's answers yourself.

##  Mini-cmp replied

You can get our collection of  Mini-cmp answers to questions from the following huggingface repository：https://huggingface.co/datasets/1f/dwa/tree/main

## Single-round display understanding dialogue assessment

```
python explicit_understanding.py
```

## Single-round display generation dialogue assessment

```
python explicit_generation.py
```

## Implicit dialogue voice scoring

```
python implicit_audio.py
```

## Implicit dialogue text scoring

```
python implicit_audio.py
```

## Implicit dialogue text scoring

```
python implicit_audio.py
```

## Multi-round dialogue voice scoring

```
python  Multi_audio.py
```

## Multi-round dialogue text scoring

```
python  Multi_text.py
```
## Attention
In the case of the multi-round dialogue evaluation, the MiniCPM model dataset above only provides a prediction of the model inference response outcome for the fourth round. The first three rounds of user-response speech need to be obtained by downloading the full WavBench dataset. The dataset will be open-sourced subsequently.
