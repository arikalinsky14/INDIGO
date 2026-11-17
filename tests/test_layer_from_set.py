"""LayerFromSet: the decode mask allows exactly the listed materials at its
position, blocks EOS there, and is silent elsewhere; check() agrees."""
import numpy as np

from inference.src.constraints import FinishedStructure, LayerFromSet, PartialStructure
from inference.src.schema import MaterialEntry
from src.materials_vocab import EOS_TOKEN, NUM_THICKNESSES


def _pool(names):
    return [MaterialEntry(canonical_name=n, n=np.ones(4), k=np.zeros(4)) for n in names]


def test_mask_and_check():
    pool = _pool(["SiO2", "Ag", "TiO2", "Au"])
    c = LayerFromSet(position=2, material_names=("Ag", "Au", "W"))
    assert c.decode_mask(PartialStructure(slots_so_far=[0], thicknesses_so_far=[50],
                                          pool_size=4), pool) is None
    m = c.decode_mask(PartialStructure(slots_so_far=[0, 2], thicknesses_so_far=[50, 50],
                                       pool_size=4), pool)
    allowed_slots = {t // NUM_THICKNESSES for t in np.flatnonzero(m) if t != EOS_TOKEN}
    assert allowed_slots == {1, 3} and not m[EOS_TOKEN]
    fs = lambda slots: FinishedStructure(slot_indices=slots, thicknesses_nm=[50] * len(slots), pool_size=4)
    assert c.check(fs([0, 2, 3]), pool)
    assert not c.check(fs([0, 2, 0]), pool)
    assert not c.check(fs([0, 2]), pool)
