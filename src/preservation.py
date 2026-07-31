"""The preservation ladder: five states of a buried bamboo-slip corpus.

Rejoining difficulty is not a free parameter of the method -- it is a property
of how much of the fracture survived burial.  Three physical quantities govern
it, and they co-vary in the ground: a corpus that has lost more of its fracture
relief has also lost more material from each face and carries more staining.

The ladder below spans well-preserved to barely legible.  Every headline result
in the paper is reported *as a function of position on this ladder*, with the
x-axis being the matcher's measured Top-k accuracy rather than the underlying
parameters, so that the findings can be read against any published matcher.
"""

from __future__ import annotations

from synth import DamageSpec

# pullout: how much distinctive relief the break carries (mm)
# face_loss: material lost independently from each of the two mating faces (mm)
# tex_noise: independent staining/consolidant corruption of each face
LADDER = [
    ("P1", "well preserved", dict(pullout_mean_mm=1.80, face_loss_mm=0.30,
                                  face_texture_noise=0.06)),
    ("P2", "good", dict(pullout_mean_mm=1.30, face_loss_mm=0.60,
                        face_texture_noise=0.14)),
    ("P3", "typical", dict(pullout_mean_mm=0.95, face_loss_mm=0.95,
                           face_texture_noise=0.24)),
    ("P4", "poor", dict(pullout_mean_mm=0.70, face_loss_mm=1.30,
                        face_texture_noise=0.38)),
    ("P5", "very poor", dict(pullout_mean_mm=0.50, face_loss_mm=1.70,
                             face_texture_noise=0.55)),
]

STATES = {k: v for k, _, v in LADDER}
LABELS = {k: lab for k, lab, _ in LADDER}


def damage(state: str, **overrides) -> DamageSpec:
    kw = dict(STATES[state])
    kw.update(overrides)
    return DamageSpec(**kw)
