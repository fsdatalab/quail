


def test_d1_2b_fp8_is_d1_2b_with_fp8_weights():
    from dataclasses import fields

    from quail.specs import D1_2B_FP8

    differ = {field.name for field in fields(D1_2B_BF16)
              if getattr(D1_2B_BF16, field.name) != getattr(D1_2B_FP8, field.name)}

    assert differ == {"name", "w_bytes", "weight_precision", "w_mem_bytes"}
    assert D1_2B_FP8.weight_precision == "fp8"
