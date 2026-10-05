import numpy as np
import torch

from wand.actions import Outcome
from wand.algebra import build_fireball
from wand.explore import SpellEnv, beam_search, map_elites
from wand.ocr import AdaptiveOCR, SymbolNet
from wand.session import WandSession
from wand.synth_data import generate_stroke_sample
from wand.wand_adapter import WandAdapter


def test_fresh_wand_is_identity():
    ad = WandAdapter()
    x = torch.randn(3, ad.effect_dim)
    poles = torch.softmax(torch.randn(3, 6), -1)
    assert torch.allclose(ad(x, poles), x)


def test_session_cast_outcome_sleep_save(tmp_path):
    s = WandSession(root=tmp_path, load_ocr=False)
    r = s.cast(build_fireball())
    assert r.program.primary_verb == "damage"
    ev = s.report_outcome(Outcome(r.program.id, totals={"damage": 10}))
    assert ev is not None and s.sleep() == 1
    s.save()
    s.close()
    s2 = WandSession(root=tmp_path, load_ocr=False)
    assert s2.mastery.rune_casts["fire"] == 1
    s2.close()


def test_adaptive_ocr_learns_without_drifting_batchnorm():
    net = SymbolNet()
    net.eval()
    before = net.block1[1].running_mean.clone()
    ocr = AdaptiveOCR(net, anchors_per_class=1)
    rng = np.random.default_rng(0)
    for _ in range(5):
        ocr.confirm(generate_stroke_sample(0, rng), "fire")
    assert torch.equal(before, net.block1[1].running_mean)
    assert ocr.feedback_from_spell([], [], {"element_intensity": [1] * 6,
                                            "domain_intensity": [0] * 6}) == 0


def test_explorer_and_env_run():
    arch = map_elites(iterations=300, init=60)
    assert len(arch.elites) > 10
    res = beam_search(lambda p: p.power, width=4, depth=2, expansions=8)
    assert res and res[0][0] > 0
    env = SpellEnv()
    env.reset()
    env.step(env.encode_action(0, 0))   # add fire
    env.step(env.encode_action(0, 13))  # add outward
    obs, reward, done, info = env.step(env.encode_action(3))
    assert done and "program" in info
