"""Chat with local LLaDA, keeping user and assistant history across turns."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time

import torch
import transformers
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = (
    "/root/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/"
    "snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
)


@torch.inference_mode()
def generate(model, prompt, length, steps, mask_id):
    """Greedily unmask the most confident tokens on a linear schedule."""
    tokens = torch.full(
        (1, prompt.shape[1] + length), mask_id,
        dtype=torch.long, device=prompt.device,
    )
    tokens[:, :prompt.shape[1]] = prompt
    for step in range(steps):
        started = time.perf_counter()
        masked = tokens == mask_id
        logits = model(tokens).logits.float()
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("Model produced nonfinite logits")
        predictions = logits.argmax(dim=-1)
        confidence = logits.softmax(dim=-1).gather(
            -1, predictions.unsqueeze(-1),
        ).squeeze(-1)
        confidence = confidence.masked_fill(~masked, -torch.inf)
        count = length // steps + (step < length % steps)
        positions = confidence.topk(count, dim=-1).indices
        tokens.scatter_(1, positions, predictions.gather(1, positions))
        print(
            f"Step {step + 1}/{steps}: {time.perf_counter() - started:.2f}s",
            flush=True,
        )
    if (tokens[:, prompt.shape[1]:] == mask_id).any().item():
        raise RuntimeError("Generation left unresolved mask tokens")
    if not torch.equal(tokens[:, :prompt.shape[1]], prompt):
        raise RuntimeError("Generation changed the input prompt")
    return tokens[:, prompt.shape[1]:]


def decode_response(tokenizer, model, token_ids):
    """Use only the answer before the first end token as the next history turn."""
    end_ids = set()
    for value in [model.config.eos_token_id, tokenizer.eos_token_id]:
        if isinstance(value, int):
            end_ids.add(value)
        elif isinstance(value, (list, tuple)):
            end_ids.update(value)
    eot_id = tokenizer.get_vocab().get('<|eot_id|>')
    if eot_id is not None:
        end_ids.add(eot_id)
    end = next((i for i, token in enumerate(token_ids) if token in end_ids), None)
    answer_ids = token_ids if end is None else token_ids[:end]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip(), end is None


def reply(model, tokenizer, messages, user_text, device, length, steps):
    """Return a completed turn without mutating history if generation fails."""
    next_messages = [*messages, {"role": "user", "content": user_text}]
    text = tokenizer.apply_chat_template(
        next_messages, tokenize=False, add_generation_prompt=True,
    )
    prompt = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    mask_id = model.config.mask_token_id
    if (prompt == mask_id).any().item():
        raise ValueError("Input contains the model mask token; remove it and try again")
    context_limit = getattr(model.config, 'max_sequence_length', None)
    if context_limit is None:
        context_limit = getattr(model.config, 'max_position_embeddings', None)
    if context_limit and prompt.shape[1] + length > context_limit:
        raise ValueError(
            f"Conversation needs {prompt.shape[1] + length} tokens; model limit is "
            f"{context_limit}. Use /reset to start a new conversation."
        )
    started = time.perf_counter()
    output = generate(model, prompt, length, steps, mask_id)
    generation_seconds = time.perf_counter() - started
    token_ids = output[0].tolist()
    response, possibly_truncated = decode_response(tokenizer, model, token_ids)
    if not response:
        raise RuntimeError("Model returned an empty response")
    return {
        "prompt": user_text, "response": response,
        "messages": [*next_messages, {"role": "assistant", "content": response}],
        "input_tokens": prompt.shape[1], "generated_token_ids": token_ids,
        "possibly_truncated": possibly_truncated,
        "generation_seconds": round(generation_seconds, 3),
    }


def save_result(path, result):
    """Replace the transcript atomically after every completed turn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", action="append", default=[], help="Initial user message; repeat for scripted turns")
    parser.add_argument("--no-interactive", action="store_true", help="Exit after the supplied --prompt turns")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device such as cuda:1")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("conversation_result.json"))
    args = parser.parse_args()
    if not 1 <= args.steps <= args.length:
        parser.error("Require 1 <= steps <= length")
    if args.threads < 1:
        parser.error("Require threads >= 1")
    if args.no_interactive and not args.prompt:
        parser.error("--no-interactive requires at least one --prompt")
    if any(not prompt.strip() for prompt in args.prompt):
        parser.error("User prompts must not be empty")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    try:
        selected_device = torch.device(device)
    except (RuntimeError, ValueError) as error:
        parser.error(f"Invalid device: {error}")
    if selected_device.type not in ['cpu', 'cuda']:
        parser.error("Use cpu, cuda, cuda:N, or auto")
    if selected_device.type == 'cuda' and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
    print(f"Loading {args.model_path} on {device} with {dtype}", flush=True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device,
    ).eval()
    load_seconds = time.perf_counter() - started
    print(f"Loaded in {load_seconds:.2f}s", flush=True)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": 0, "temperature": 0,
        "messages": [], "turns": [], "conversation_index": 0,
        "generation_length": args.length,
        "diffusion_steps": args.steps,
        "load_seconds": round(load_seconds, 3),
        "status": "chatting",
    }
    messages = []
    pending = iter(args.prompt)
    print("Chat ready. Commands: /reset (clear history), /exit or /quit (finish).", flush=True)
    print(f"Transcript: {args.output}", flush=True)
    try:
        while True:
            user_text = next(pending, None)
            scripted = user_text is not None
            if not scripted:
                if args.no_interactive:
                    break
                try:
                    user_text = input("You: ").strip()
                except EOFError:
                    break
                if not user_text:
                    continue
                if user_text.lower() in ['/exit', '/quit']:
                    break
                if user_text.lower() == '/reset':
                    messages = []
                    result['messages'] = []
                    result['conversation_index'] += 1
                    save_result(args.output, result)
                    print("Conversation history cleared.", flush=True)
                    continue
            else:
                print(f"You: {user_text}", flush=True)
            try:
                turn = reply(model, tokenizer, messages, user_text, device, args.length, args.steps)
            except ValueError as error:
                if scripted:
                    raise
                print(f"Cannot answer: {error}", flush=True)
                continue
            messages = turn.pop('messages')
            turn['conversation_index'] = result['conversation_index']
            result['messages'] = messages
            result['turns'].append(turn)
            result['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
            save_result(args.output, result)
            print(f"LLaDA: {turn['response']}", flush=True)
            if turn['possibly_truncated']:
                print("[Output reached the token limit; increase --length/--steps for longer replies.]", flush=True)
    except KeyboardInterrupt:
        result['status'] = 'interrupted'
        print("\nChat interrupted; completed turns saved.", flush=True)
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    else:
        result['status'] = 'completed'
    finally:
        result['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        save_result(args.output, result)


if __name__ == "__main__":
    main()
