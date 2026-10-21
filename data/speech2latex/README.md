Speech2Latex (marsianin500/Speech2Latex), English equations only. TRAIN SPLIT IS INTENTIONALLY ABSENT: this project fine-tunes nothing, so no training data is used from this dataset.
- en_dev (1,086 clips): tuning ONLY — β (acoustic mix), LLM prompts, normalisation rules. Never report results on it.
- en_test_500 (500 clips: 270 human, 230 synthetic): MAIN reported benchmark for ablations A–D.
- en_test (2,745 clips): one extended final run if time permits; otherwise untouched.
No example from en_test/en_test_500 may be used for tuning or prompt examples.
