"""E09a: B0 sonar-only under the fixed training recipe, NO scale jitter.

Isolates the recipe axis (review 2026-10-31 §1.4: COCO-pretrained init,
YOLOX lr scaling, no norm/bias decay, quadratic warmup + cosine + tail,
EMA).  E07's B0 and E08's B0 both trained under the old recipe (8x lr,
no pretrain, no EMA), so the recording-domain collapse could never be
attributed apart from overfitting; this arm is the recipe-only control
for the pre-registered E09 criteria (doc/experiments.md E09).
"""

_base_ = "s0_b0_sonar_only_yoloxs.py"

# no train_pipeline override: identical to the s0_b0 base (no
# AOSonarScaleJitter).  The file exists to pin the E09a provenance and
# keep the run reproducible against later edits of the base config.
