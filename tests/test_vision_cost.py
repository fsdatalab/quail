"""CPU checks for the vision tower in the speed of light estimate."""

import pytest

from quail.cost.sol import speed_of_light
from quail.cost.vision_cost import image_work, vision_components
from quail.cost.work import Work
from quail.pdf import PDFInput, PdfPageRef, PdfRowRef, PdfSource
from quail.planner.estimate import image_page_sizes
from quail.specs import DIFFUSION_GEMMA_26B_FP8, H100_SXM, QWEN3_4B_FP8

GEMMA = DIFFUSION_GEMMA_26B_FP8
# Patch projection, the 2D position table, 27 layers of attention and
# a gated MLP, and the projection into the decoder width. Without that
# last projection this is 569.4M, against the 569.6M encoder figure,
# which also counts the tower's norms.
TOWER_PARAMS = 572_663_808


def test_one_page_prices_the_tower_and_keeps_attention_inside_it():
    work = image_work([(612, 792), (612, 792)], GEMMA, 280)
    assert work.image_pairs == 2 * (work.image_patches / 2) ** 2
    result = speed_of_light(work, GEMMA, H100_SXM, 65_536)
    names = [component.name for component in result.components]
    assert names[:5] == [
        "vision_embed", "vision_attn_proj", "vision_mlp",
        "vision_attention", "vision_project"]
    weight_bytes = sum(
        component.bytes_moved for component in result.components
        if component.name != "vision_attention")
    assert weight_bytes == TOWER_PARAMS * GEMMA.vision_tower.weight_bytes
    assert result.component("vision_mlp").flops == (
        2 * 3 * GEMMA.vision_tower.hidden * GEMMA.vision_tower.intermediate
        * GEMMA.vision_tower.layers * work.image_patches)
    assert all(component.precision == "bf16" for component in result.components
               if component.name.startswith("vision_"))
    # the tower's compute is the bound; its weights are one read
    assert result.component("vision_attention").bound_by == "compute"
    assert result.seconds > speed_of_light(
        Work(), GEMMA, H100_SXM, 65_536).seconds


def test_no_pages_and_a_text_model_add_no_tower_components():
    assert image_work([], GEMMA, 280) == Work()
    assert vision_components(Work(), GEMMA) == ()
    text = speed_of_light(Work(tokens=100, pairs=100), QWEN3_4B_FP8,
                          H100_SXM, 1000)
    assert [component.name for component in text.components] == [
        "attn_proj", "mlp", "attention"]
    with pytest.raises(ValueError, match="no vision tower"):
        vision_components(
            Work(image_patches=4, image_pairs=16, image_soft_tokens=1),
            QWEN3_4B_FP8)


def test_image_counts_take_part_in_dominance():
    plain = Work(tokens=10)
    imaged = Work(tokens=10, image_patches=4, image_pairs=16,
                  image_soft_tokens=1)
    assert plain.dominates(imaged)
    assert not imaged.dominates(plain)
    assert imaged + imaged == imaged * 2


def test_a_page_named_twice_or_by_two_inputs_is_embedded_once():
    source = PdfSource("a.pdf", 10, 20)
    page = PdfPageRef(0, 0, 612, 792)
    repeated = PDFInput(
        sources=(source,), pages=(page,),
        rows=(PdfRowRef((0,)), PdfRowRef((0,))), row_mode="page")
    other = PDFInput(
        sources=(source,), pages=(page,),
        rows=(PdfRowRef((0,)),), row_mode="page")
    assert image_page_sizes([repeated], credit_shared=False) == [(612, 792)]
    shared = image_page_sizes([repeated, other], credit_shared=True)
    separate = image_page_sizes([repeated, other], credit_shared=False)
    assert shared == [(612, 792)]
    assert separate == [(612, 792), (612, 792)]
