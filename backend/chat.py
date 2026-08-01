#!/usr/bin/env python3
"""
Local chat / transcript-analysis LLM for the AI Analysis panel.

Backends:
  - "mlx"          Qwen3-4B-Instruct (MLX 4-bit) on the Apple-Silicon GPU.
                   Only importable on macOS + Apple Silicon.
  - "transformers" The same Qwen3-4B-Instruct model via HF `transformers`,
                   run on an NVIDIA GPU (float16) if available, else CPU.
                   Cross-platform: this is the path used on Windows/Linux.

Selection is automatic ("auto"): tries mlx first, falls back to transformers.
Force one explicitly with the CHAT_BACKEND env var ("mlx" | "transformers")
if the auto-detection ever guesses wrong for your machine.

Both backends expose the same public functions used by server.py:
stream_chat(messages, transcript) and make_title(transcript).
"""
import glob
import os
import sys
import threading

MLX_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
# Windows/Linux intentionally run a DIFFERENT (newer) chat model than the Mac
# MLX path: Qwen3.5-4B, not Qwen3-4B-Instruct-2507. That is a deliberate
# choice, not a mismatch to "fix" back to matching MLX_MODEL.
#
# Qwen3.5-4B's own config.json declares architectures:
# ["Qwen3_5ForConditionalGeneration"] (the vision-language class) plus a full
# vision_config -- but that field is metadata, not what AutoModelForCausalLM
# actually loads: transformers' MODEL_FOR_CAUSAL_LM_MAPPING registers
# "qwen3_5" -> Qwen3_5ForCausalLM, a dedicated text-only class
# (_keys_to_ignore_on_load_unexpected = ["^model.visual.*", "^mtp.*"]) that
# never instantiates the vision tower. AutoModelForCausalLM.from_pretrained()
# below is therefore already correct -- do not change it to a vision/VLM auto
# class, and do not assume the vision weights cost anything at inference time.
#
# What genuinely needs care with this model (see _ensure() and stream_chat()):
#   1. It requires a transformers version that registers "qwen3_5" -- per its
#      own model card, that means installing from the `main` branch, not a
#      pip release pinned by a `>=` floor. An unsupporting transformers fails
#      the FIRST from_pretrained() call outright.
#   2. It thinks by default (emits a <think>...</think> block before every
#      reply, recommended up to 32k-80k tokens for hard tasks per its model
#      card) unless enable_thinking=False is honoured by the chat template.
#      If that silently stops working, every reply pays for a full hidden
#      reasoning pass -- which reads exactly like "processing got slower".
HF_MODEL = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen3.5-4B")

# "qwen3_5" landing in a stable transformers release lags this model's launch;
# many deployments will only have it via `pip install
# "transformers @ git+https://github.com/huggingface/transformers.git@main"`
# (see the model card). Checked once in _ensure() so a mismatch fails with an
# actionable message instead of a bare "Unrecognized configuration class"
# traceback from deep inside from_pretrained().
_REQUIRED_MODEL_TYPE = "qwen3_5"
_SNAP_GLOB = "models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/chat_template.jinja"

_active = None  # "mlx" | "transformers"
_MODEL = None
_TOK = None
_TMPL = None
# Diagnostic snapshot of what _ensure() actually loaded -- see backend_info().
_device_info: dict = {}


def _try_mlx() -> bool:
    try:
        import mlx_lm  # noqa: F401 -- availability probe only
        return True
    except Exception:
        return False


def _select():
    global _active
    if _active is not None:
        return
    requested = os.environ.get("CHAT_BACKEND", "auto").lower()
    if requested in ("mlx", "auto") and _try_mlx():
        _active = "mlx"
    else:
        _active = "transformers"


def active_model_name() -> str:
    _select()
    return MLX_MODEL if _active == "mlx" else HF_MODEL


def _ensure():
    """Lazily load the active backend's model/tokenizer (and MLX chat
    template, if applicable)."""
    global _MODEL, _TOK, _TMPL
    _select()
    if _MODEL is not None:
        return _MODEL, _TOK, _TMPL
    if _active == "mlx":
        from mlx_lm import load
        _MODEL, _TOK = load(MLX_MODEL)
        hits = glob.glob(os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), _SNAP_GLOB))
        _TMPL = open(hits[0]).read() if hits else None
        _device_info.update(backend="mlx", model=MLX_MODEL, device="metal", quantization="4bit")
    else:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Only probe for qwen3_5 support when HF_MODEL actually looks like a
        # Qwen3.5-family checkpoint -- HF_CHAT_MODEL can be pointed at anything,
        # and this check would otherwise misfire on an unrelated model.
        looks_qwen35 = "qwen3.5" in HF_MODEL.lower() or "qwen3_5" in HF_MODEL.lower()
        supported = None
        if looks_qwen35:
            try:
                from transformers.models.auto.configuration_auto import CONFIG_MAPPING
                supported = _REQUIRED_MODEL_TYPE in CONFIG_MAPPING
            except Exception:
                pass  # couldn't even check -- let from_pretrained() speak for itself
        if supported is False:
            raise RuntimeError(
                f"installed transformers ({transformers.__version__}) does not "
                f"recognize model_type={_REQUIRED_MODEL_TYPE!r}, which {HF_MODEL} "
                "requires. This model shipped ahead of a stable transformers "
                "release that supports it -- install from main:\n"
                '  pip install "transformers[serving] @ '
                'git+https://github.com/huggingface/transformers.git@main"\n'
                "(also needs torchvision and pillow present, even for text-only "
                "use -- see the model card's Transformers quickstart). Or pin "
                "HF_CHAT_MODEL to a model your installed transformers supports.")
        # CHAT_DEVICE forces cpu on a GPU box: Qwen3.5-4B is 4.66B params, so
        # ~9GB in fp16 -- over half a 16GB card held for a panel that is used
        # on demand. CPU generation is much slower but leaves the GPU entirely
        # to ASR/diarization.
        #
        # fp16 (not bf16) on CUDA is deliberate: the checkpoint is bf16, but
        # bf16 needs Ampere (sm_80+) and this deploys to a T4 (Turing, sm_75),
        # where bf16 has no hardware support while fp16 has full tensor-core
        # support.
        want = os.environ.get("CHAT_DEVICE", "auto").lower()
        if want in ("cuda", "cpu"):
            device = want
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _TOK = AutoTokenizer.from_pretrained(HF_MODEL)

        # Quantization is OFF by default: the model runs at full fp16 on CUDA.
        #
        # This used to default to bitsandbytes 8-bit, which was a memory
        # decision that quietly cost latency. LLM.int8() is a footprint
        # optimization, not a speed one -- it dequantizes on the fly and runs a
        # mixed-precision decomposition for outlier channels, and it only pays
        # for itself on large hidden dimensions. Below that crossover it is
        # SLOWER than plain fp16, and reported figures are not subtle (a 7B on
        # an A100: ~6.7 tok/s at 8-bit vs ~16.7 tok/s at fp16). Qwen3.5-4B's
        # hidden size is 2560, i.e. squarely on the wrong side of that
        # crossover, so 8-bit here was buying VRAM with generation speed.
        #
        # CHAT_8BIT=1 opts back in when VRAM is the binding constraint --
        # roughly halves the footprint (~9GB fp16 -> ~5GB). Note the chat model
        # is unloaded before every transcription (see server.py), so it only
        # contends with ASR/diarization if a chat request arrives mid-job;
        # CHAT_DEVICE=cpu is the other escape hatch.
        #
        # Quantized loads must NOT be followed by .to(device): bitsandbytes
        # places the weights itself through accelerate, and moving the module
        # afterwards raises. Hence the separate device_map path below.
        quant_cfg = None
        if device == "cuda" and os.environ.get("CHAT_8BIT", "0") != "0":
            try:
                import bitsandbytes  # noqa: F401 -- availability probe only
                from transformers import BitsAndBytesConfig
                quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                print(f"[chat] 8-bit requested but unavailable ({type(e).__name__}: {e}) -- "
                      "falling back to fp16. Install bitsandbytes to halve the "
                      "chat model's VRAM.", file=sys.stderr, flush=True)

        if quant_cfg is not None:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                HF_MODEL, quantization_config=quant_cfg, device_map={"": 0})
            print(f"[chat] {HF_MODEL} loaded in 8-bit on cuda", file=sys.stderr, flush=True)
            _device_info.update(backend="transformers", model=HF_MODEL, device="cuda",
                                quantization="8bit")
        else:
            _MODEL = AutoModelForCausalLM.from_pretrained(
                HF_MODEL, torch_dtype=dtype).to(device)
            print(f"[chat] {HF_MODEL} loaded in {dtype} on {device}", file=sys.stderr, flush=True)
            _device_info.update(backend="transformers", model=HF_MODEL, device=device,
                                quantization=None)
        _MODEL.eval()
    return _MODEL, _TOK, _TMPL


def backend_info() -> dict:
    """Diagnostic snapshot of which chat model/device/quantization actually
    got loaded -- call _ensure() (or send one chat message) first if this
    returns {"loaded": False}; deliberately does not force a load itself."""
    if _MODEL is None:
        return {"loaded": False}
    return {"loaded": True, **_device_info}


def unload() -> bool:
    """Drop the chat model and free its GPU memory. Returns True if something
    was actually unloaded.

    _ensure() caches the model for the process lifetime, so once the AI Analysis
    panel has been used once, ~8GB stays occupied for every subsequent
    transcription -- which is what pushes long files into CUDA OOM. Transcription
    calls this first; the next chat request reloads lazily (a few seconds).

    A generation already in flight keeps its own reference, so this is safe to
    call concurrently -- that memory just frees when the stream finishes."""
    global _MODEL, _TOK, _TMPL
    if _MODEL is None:
        return False
    _MODEL = _TOK = _TMPL = None
    _device_info.clear()
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return True


def _system_prompt(transcript: str) -> str:
    if transcript:
        return (
            "You are a speech analytics AI assistant. You have access to the following "
            "transcript from a recorded conversation. Each line is prefixed with the "
            "speaker and the exact timestamp it was said, e.g. \"[S1 04:12]: ...\".\n\n"
            "TRANSCRIPT:\n" + transcript +
            "\n\nGuidelines:\n"
            "- Always respond in Persian (Farsi) regardless of the question's language\n"
            "- Keep answers brief and to the point\n"
            "- Reference specific speakers (S1, S2, S3, …) when relevant\n"
            "- When you state something the transcript says, cite EXACTLY where by copying "
            "that line's full bracket verbatim, e.g. [S1 04:12] -- copy it exactly as it "
            "appears in the transcript above, speaker letter included; do not estimate, "
            "reformat, or drop the speaker. This lets the user jump to and "
            "verify that moment, so include one whenever you reference a specific claim, "
            "decision, or quote -- not for general summaries with no single source line"
        )
    return ("You are a helpful AI assistant specialized in speech transcription and audio "
            "analysis. Always respond in Persian (Farsi) briefly and to the point.")


def _build_messages(messages, transcript):
    msgs = [{"role": "system", "content": _system_prompt(transcript)}]
    for m in messages:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def _context_window(model, tok) -> int:
    """Best-guess max sequence length the loaded model actually supports.
    transformers configs vary in which attribute carries this, and an unset
    tokenizer.model_max_length is a ~1e30 sentinel rather than a real number --
    both need guarding against, or the fallback below is silently never used."""
    ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if ctx and ctx < 1_000_000:
        return ctx
    mm = getattr(tok, "model_max_length", None)
    if mm and mm < 1_000_000:
        return mm
    return 32768


def _fit_to_context(tok, messages, transcript, max_new_tokens, max_ctx):
    """Shrink transcript then drop the oldest chat turns until the rendered
    prompt fits max_ctx - max_new_tokens. The full transcript is re-embedded
    in the system prompt on EVERY turn (see _system_prompt), so total tokens
    grow with both conversation length and transcript length -- on a long
    recording, a handful of follow-up questions is enough to exceed the
    model's context window. Whether that then errors or just produces
    garbage depends on the model, but either way the fix is the same: keep
    the rendered prompt under the ceiling before generating, rather than
    discovering the overflow from a failed generate() call."""
    def render(msgs):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)

    budget = max_ctx - max_new_tokens - 64
    msgs = _build_messages(messages, transcript)
    try:
        if len(render(msgs)) <= budget:
            return msgs
    except Exception:
        return msgs  # template/tokenizer surprised us -- let generate() surface it directly

    # 1) shrink the transcript first -- usually the largest single contributor,
    # and cutting it is less disruptive than dropping conversation turns
    t = transcript
    while t:
        t = t[: int(len(t) * 0.7)]
        msgs = _build_messages(messages, t)
        try:
            if len(render(msgs)) <= budget or not t:
                break
        except Exception:
            break

    # 2) still too long (a long chat history even with no transcript): drop
    # the oldest turns, always keeping at least the latest user message
    trimmed = list(messages)
    while len(trimmed) > 1:
        msgs = _build_messages(trimmed, t)
        try:
            if len(render(msgs)) <= budget:
                break
        except Exception:
            break
        trimmed.pop(0)
    return _build_messages(trimmed, t)


def _check_thinking_leak(tok, output_ids, prompt_len):
    """Confirm enable_thinking=False actually suppressed Qwen3.5's default
    <think>...</think> reasoning block -- and say so loudly if it didn't.

    If this silently stops working (a template change, a version mismatch
    between the cached chat_template.jinja and the installed transformers),
    every reply pays for a full hidden reasoning pass -- the model card
    recommends up to 32k-80k tokens for hard tasks -- while
    TextIteratorStreamer's skip_special_tokens=True means the client-visible
    stream shows nothing wrong: no error, no visible <think> tag, just a
    reply that takes much longer to arrive. That is precisely "the chat got
    slower, no error, nothing visibly different" from the outside, so this
    decodes the raw output once per response (cheap relative to the
    generation that already happened) purely to make that failure mode loud
    instead of invisible."""
    try:
        gen_ids = output_ids[0][prompt_len:]
        text = tok.decode(gen_ids, skip_special_tokens=False)
    except Exception:
        return
    if "<think>" not in text:
        return
    body = text.split("<think>", 1)[1]
    body = body.split("</think>", 1)[0] if "</think>" in body else body
    try:
        n = len(tok.encode(body, add_special_tokens=False)) if body else 0
    except Exception:
        n = None
    print(f"[chat] enable_thinking=False did NOT suppress a <think> block "
         f"({'~' + str(n) if n is not None else 'unknown'} hidden tokens "
         "generated) -- this response paid for a full reasoning pass that "
         "never reaches the client. Suspect a chat_template.jinja / "
         "transformers version mismatch; see Qwen3.5's model card, "
         "'Instruct (or Non-Thinking) Mode'.", file=sys.stderr, flush=True)


def stream_chat(messages, transcript="", max_tokens=1024, temperature=0.7):
    """Yield generated Persian text token-by-token, on whichever backend is active."""
    model, tok, tmpl = _ensure()

    if _active == "mlx":
        msgs = _build_messages(messages, transcript)
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, chat_template=tmpl)
        sampler = make_sampler(temp=temperature)
        for resp in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler):
            if resp.text:
                yield resp.text
        return

    # transformers backend: generate in a background thread, stream via TextIteratorStreamer
    import torch
    from transformers import TextIteratorStreamer

    msgs = _fit_to_context(tok, messages, transcript, max_tokens, _context_window(model, tok))
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, streamer=streamer,
                      do_sample=temperature > 0, temperature=max(temperature, 0.01))

    # generate() runs in its own thread so the streamer can be consumed here as
    # tokens arrive. Without the try/except, an exception in that thread (context
    # overflow, a transient CUDA OOM while ASR/diarization also hold the GPU) is
    # printed to stderr by Python's default thread excepthook and otherwise
    # vanishes -- crucially, streamer.end() is never called, so `for text in
    # streamer` below blocks forever waiting for a token that will never come.
    # That is the hang this project's users hit as "stops responding after a
    # few questions, have to start a new conversation": the request never
    # completes, so there is nothing for the client to time out on either.
    error: list = []
    raw_output: list = []

    def _run():
        try:
            raw_output.append(model.generate(**gen_kwargs))
        except Exception as e:
            # Store the MESSAGE, not the exception object. An exception keeps
            # its __traceback__, the traceback keeps every frame, and those
            # frames keep the activations and KV cache that just failed to fit.
            # Holding it here would pin exactly the GPU memory we are trying to
            # recover from -- so a CUDA OOM would make the next attempt more
            # likely to OOM, not less.
            error.append(f"{type(e).__name__}: {e}")
            streamer.end()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for text in streamer:
        if text:
            yield text
    thread.join()
    if error:
        # drop references to the failed generation's tensors before raising,
        # so the caller's recovery (unload + empty_cache) can actually reclaim
        del gen_kwargs, inputs
        raw_output.clear()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        raise RuntimeError(f"generation failed: {error[0]}")
    if raw_output:
        _check_thinking_leak(tok, raw_output[0], inputs["input_ids"].shape[-1])


def make_title(transcript: str) -> str:
    model, tok, tmpl = _ensure()
    msgs = [{"role": "user", "content":
             "Generate a concise 4-6 word Persian title for this transcript "
             "(no quotes, no trailing punctuation):\n\n" + transcript[:500]}]

    if _active == "mlx":
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, chat_template=tmpl)
        out = generate(model, tok, prompt=prompt, max_tokens=24,
                       sampler=make_sampler(temp=0.5), verbose=False).strip()
    else:
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
        out_ids = model.generate(**inputs, max_new_tokens=24, do_sample=True, temperature=0.5)
        out = tok.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    return out.splitlines()[0].strip('"“”') if out else "تحلیل جدید"
