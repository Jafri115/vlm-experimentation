#!/usr/bin/env python
from pathlib import Path
import shutil
import sys

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "./scripts/run_qwen3vl_fewshot_text_balanced100.py"
)

if not target.exists():
    raise FileNotFoundError(target)

text = target.read_text(encoding="utf-8")

old = """        raw = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        result = parse_text_classification(raw)

        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result, raw, inference_sec
"""

new = """        raw = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Normal parse first. If the model emits an invalid task label
        # (for example WITHDRAWAL), do not map it manually. Ask the same
        # model to convert its previous decision into the allowed binary
        # output space.
        try:
            result = parse_text_classification(raw)

        except Exception:
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            repair_prompt = (
                "Your previous answer for this binary classification task "
                "used an invalid label.\\n\\n"
                "Previous answer:\\n"
                + raw
                + "\\n\\nAllowed labels are ONLY:\\n"
                "- RUPTURE\\n"
                "- NO_RUPTURE\\n\\n"
                "Do not reconsider the evidence and do not introduce a "
                "third category such as WITHDRAWAL or CONFRONTATION. "
                "Convert your previous decision into the single closest "
                "allowed binary label.\\n\\n"
                "Return exactly one of these strings:\\n"
                "RUPTURE\\n"
                "NO_RUPTURE"
            )

            repair_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": repair_prompt,
                        }
                    ],
                }
            ]

            repair_chat_text = self.processor.apply_chat_template(
                repair_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            repair_inputs = self.processor(
                text=[repair_chat_text],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            repair_started = time.time()

            with torch.inference_mode():
                repair_ids = self.model.generate(
                    **repair_inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    use_cache=True,
                )

            inference_sec += time.time() - repair_started

            repair_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    repair_inputs.input_ids,
                    repair_ids,
                )
            ]

            repair_raw = self.processor.batch_decode(
                repair_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            try:
                result = parse_text_classification(repair_raw)
            except Exception as repair_exc:
                raise ValueError(
                    "Invalid binary label after format repair. "
                    f"Initial output={raw!r}; "
                    f"repair output={repair_raw!r}"
                ) from repair_exc

            raw = raw + "\\n\\n[FORMAT_REPAIR]\\n" + repair_raw

            del repair_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if "inputs" in locals():
            try:
                del inputs
            except Exception:
                pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result, raw, inference_sec
"""

if old not in text:
    if "[FORMAT_REPAIR]" in text:
        print("Patch already appears to be installed.")
        sys.exit(0)

    raise RuntimeError(
        "Expected classify_description_text block was not found. "
        "No changes were made."
    )

backup = target.with_suffix(target.suffix + ".bak")
shutil.copy2(target, backup)

patched = text.replace(old, new, 1)

# Critical safety check: only write if the patched target is valid Python.
compile(patched, str(target), "exec")

target.write_text(patched, encoding="utf-8")

print(f"Patched: {target}")
print(f"Backup:  {backup}")