"""E11 M1 arm: capacity-only distillation -- the SAME big clear-trained
fusion teacher as M3, but seeing the student's degraded camera
(teacher_input='degraded'), so no privileged information flows (F2 §2.2
M1, the capacity cell of the 2x2).

M3 - M1 isolates the privilege effect, M1 - D-self the capacity effect
(F2 §6.3).  Base is e10_m3_privileged_yoloxs itself and the ONLY
override is teacher_input, so M1-vs-M3 is exactly clear-vs-degraded
teacher view at equal checkpoint / tau / routes / weights.  tau stays 2:
the tau spectrum is E10 criterion 6's axis, not E11's.
"""

_base_ = ["e10_m3_privileged_yoloxs.py"]

model = dict(
    distillation=dict(
        teacher_input="degraded",
    ),
)
