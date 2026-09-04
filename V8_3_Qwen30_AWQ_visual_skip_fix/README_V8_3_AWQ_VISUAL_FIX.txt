V8.3 — Qwen3-VL-30B-A3B AWQ visual-tower skip fix

V8.2 error:
AwqGEMMTritonLinear: in_features 4304 must be divisible by group_size 128.

Cause:
- Qwen3-VL vision_config intermediate_size is 4304.
- QuantTrio checkpoint was quantized with the visual tower ignored.
- Its config records modules_to_not_convert=["visual", "mlp.gate"].
- Transformers 5.15 skip matching does not make bare "visual" match the nested
  module path "model.visual.*".
- As a result Transformers tried to construct AWQ layers inside the FP16 visual
  tower, which was never intended to be quantized.

V8.3 changes ONLY the runtime config skip list to:
["visual", "mlp.gate", "model.visual"]

It does not edit the cached checkpoint and does not requantize anything.

Run first:
powershell -ExecutionPolicy Bypass -File .\test_v8_3_qwen30_awq_load.ps1

If it says LOAD SUCCESS and VISUAL SKIP CHECK: PASS, run:
powershell -ExecutionPolicy Bypass -File .\run_v8_3_qwen30_awq_smoke_5.ps1
