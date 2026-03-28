"""
examples/huggingface_integration.py
────────────────────────────────────
Monitor a real local HuggingFace model.
Requires: pip install torch transformers

Run:  python examples/huggingface_integration.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rgi_monitor import LLMAdapter, TruthSignal, UniversalMonitor

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("torch/transformers not installed — running in simulation mode")
    print("Install with: pip install rgi-monitor[hf]")


def get_logprob(model, tokenizer, text: str, device: str) -> float:
    """Compute mean log-probability of text under the model."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return -float(outputs.loss)   # negative loss = log-prob


def main():
    model_name = "Qwen/Qwen1.5-0.5B"   # small, fast, open-weights
    device     = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

    mon = UniversalMonitor(agent_id=model_name, verbose=True)

    prompts = [
        "The capital of France is",
        "Water boils at 100 degrees",
        "The moon is made of cheese",   # hallucination-prone prompt
        "Python is a programming language",
        "Gravity pulls objects toward each other",
    ]

    if HAS_TORCH:
        print(f"  Loading {model_name} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        print(f"  Model loaded. Running {len(prompts)} prompts...\n")

        for prompt in prompts:
            # Score the prompt alone (before generation)
            logprob_before = get_logprob(model, tokenizer, prompt, device)

            # Generate a short continuation
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=20,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Score the full text (prompt + generation)
            logprob_after = get_logprob(model, tokenizer, full_text, device)

            # Convert to R1
            R1 = LLMAdapter.from_logprob_delta(logprob_before, logprob_after)

            # Build truth signal from logprob-derived surprisal
            # In production: use RAG retrieval + citation counts
            surprisal = max(0.0, min(1.0, -logprob_after / 10.0))
            truth_sig = TruthSignal(
                evidence     = 0.7 if surprisal < 0.3 else 0.3,
                surprisal    = surprisal,
                sources      = 1,
                contradiction= 0.0,
            )

            snap = mon.step(R1=R1, truth_signal=truth_sig)
            print(f"  Prompt: {prompt!r}")
            print(f"  Output: {full_text[len(prompt):].strip()!r}")
            print()

    else:
        # Simulation mode — runs without torch
        import random
        random.seed(99)
        print(f"  Simulation mode (no torch). Running {len(prompts)} turns...\n")
        for i, prompt in enumerate(prompts):
            R1       = random.gauss(0.03, 0.03)
            truth_sig = TruthSignal(
                evidence=random.uniform(0.5, 0.95),
                surprisal=random.uniform(0.05, 0.4),
                sources=random.randint(1, 4),
                contradiction=random.uniform(0.0, 0.15),
            )
            snap = mon.step(R1=R1, truth_signal=truth_sig)
            print(f"  Simulated prompt: {prompt!r}")
            print()

    print("  Final report:")
    print(mon.export_json())


if __name__ == "__main__":
    main()
