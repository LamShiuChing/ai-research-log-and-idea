# Test-time memory chat app (Titans NeuralMemory on a frozen LM)
**Dates:** source files last modified Sep 5 (no date stated in the source) · **Status:** built, not evaluated · **Hardware/scale:** RTX 4080 SUPER 16 GB, Windows 11, Python 3.10.11 (as stated in the README)

## Question / goal
Can a small Titans-style neural memory, written to and trained at test time, let a local chat model recall facts from earlier conversation after the chat context is cleared, while the base model's weights stay frozen?

## Approach
- Local desktop chat app: PyQt6 GUI, one worker thread running all PyTorch work, tokens streamed back to the GUI through a Qt signal.
- Base model: any Hugging Face causal LM with a chat template and full attention (examples in the README: Qwen3-4B, SmolLM3-3B, Qwen2.5-0.5B-Instruct).
- Memory: a `NeuralMemory` module from `titans-pytorch` (a small MLP whose weights act as the memory, updated by surprise gradients with momentum and forgetting), attached to one decoder layer as a zero-initialised gated residual: `h_out = h + gate * memory.retrieve(h)`. With gate = 0 the base model is unchanged.
- Inner (Titans) updates: memory weights are updated every `chunk_size` tokens from hidden states, with a learned adaptive step capped by "Inner max step".
- Outer TTT step: after each assistant turn, the turn is re-run with the memory in differentiable mode and the LM loss is backpropagated only into the memory module, using AdamW. Base weights are never written.
- Commit: the turn is written into memory once and the KV cache is rebuilt, so each token is stored once.
- Persistence: save/load of memory parameters, inner memory state, and transcript under a workspace folder. "Forget memory contents" clears inner state but keeps learned parameters.
- Libraries: torch, transformers, titans-pytorch, tensordict, huggingface_hub, PyQt6, Markdown, einops.

## Results
Not evaluated yet. The repo contains no benchmark, metric, or recall test results. The README describes a manual check (tell the model facts, start a new chat with memory kept, ask about them) but no outcome of that check is recorded.

## Limitations
- No measured results; claims about memory recall are untested in this repo.
- Models with sliding-window attention layers (Gemma-2/3) are not supported by the TTT step because their KV cache cannot be rolled back.
- MPS (Apple) support is listed in the README as untested.
- The first message after a model load or change re-prefills the whole transcript.
- Memory size scales with `heads` x `head_dim`; a full-width memory on a 4B model needs tens of GB during the TTT step, so the app uses a narrower default (4 x 128).
- Requires a CUDA build of torch on NVIDIA machines; install steps differ by platform.

## Key files
- `README.md`: architecture, install steps, model list, usage walkthrough, and knob descriptions.
- `main.py`: main window, worker-thread wiring, entry point (`python main.py`).
- `worker.py`: `ModelWorker`, a QObject that runs model load, generation, TTT updates, and memory save/load on its own thread and reports through Qt signals.
- `model_engine.py`: `ChatEngine`, `TitansAdapter` (the gated memory residual), `MemoryConfig` / `GenConfig`, and device/precision selection.
- `ui_components.py`: `ChatView` (streaming Markdown and code blocks) and `ControlPanel` (Titans, TTT, and sampling settings).
- `requirements.txt`: pinned package versions (torch 2.14.0, transformers 5.16.1, titans-pytorch 0.5.5, PyQt6 6.11.0, and others).

## Sources
- Titans paper (arXiv 2501.00663), cited in the README as the basis for the neural long-term memory.
- `titans-pytorch` package (NeuralMemory implementation) and Hugging Face `transformers`.
- Project README and source docstrings in the TTT app folder.
