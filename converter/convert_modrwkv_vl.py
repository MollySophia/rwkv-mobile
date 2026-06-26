#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_safetensors(model_dir: Path) -> Iterable[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_json(index_path)
        seen = sorted(set(index.get("weight_map", {}).values()))
        for filename in seen:
            yield model_dir / filename
        return

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    yield from files


def _map_llm_key(key: str) -> str | None:
    if key == "model.llm.embeddings.weight":
        return "emb.weight"
    if key == "lm_head.weight":
        return "head.weight"
    if key.startswith("model.llm.norm."):
        return key.replace("model.llm.norm.", "ln_out.", 1)

    prefix = "model.llm.layers."
    if not key.startswith(prefix):
        return None

    rest = key[len(prefix):]
    layer, sep, name = rest.partition(".")
    if not sep or not layer.isdigit():
        return None

    block = f"blocks.{layer}."
    replacements = {
        "pre_norm.": "pre_ln.",
        "attn_norm.": "ln1.",
        "ffn_norm.": "ln2.",
        "ffn.": "ffn.",
        "attn.g_norm.": "att.ln_x.",
        "attn.k_proj.": "att.key.",
        "attn.v_proj.": "att.value.",
        "attn.r_proj.": "att.receptance.",
        "attn.o_proj.": "att.output.",
        "attn.w_lora.lora.0.": "att.w1.",
        "attn.w_lora.lora.2.": "att.w2.",
        "attn.a_lora.lora.0.": "att.a1.",
        "attn.a_lora.lora.2.": "att.a2.",
        "attn.v_lora.lora.0.": "att.v1.",
        "attn.v_lora.lora.2.": "att.v2.",
        "attn.g_lora.lora.0.": "att.g1.",
        "attn.g_lora.lora.2.": "att.g2.",
        "attn.": "att.",
    }
    for src, dst in replacements.items():
        if name.startswith(src):
            mapped = block + name.replace(src, dst, 1)
            break
    else:
        return None

    # FLA LoRA modules store the additive base as the second linear bias.
    mapped = mapped.replace(".w2.bias", ".w0.weight")
    mapped = mapped.replace(".a2.bias", ".a0.weight")
    mapped = mapped.replace(".v2.bias", ".v0.weight")

    # In standard RWKV Eagle .pth format, block 0 uses ln0 (not pre_ln)
    # as the pre-attention layer norm.
    if mapped.startswith("blocks.0.pre_ln."):
        mapped = "blocks.0.ln0." + mapped[len("blocks.0.pre_ln."):]

    # Standard RWKV .pth stores lora-style weight keys without .weight suffix.
    lora_weight_keys = (
        ".att.w1.weight", ".att.w2.weight",
        ".att.a1.weight", ".att.a2.weight",
        ".att.v1.weight", ".att.v2.weight",
        ".att.g1.weight", ".att.g2.weight",
        ".att.w0.weight", ".att.a0.weight", ".att.v0.weight",
    )
    if mapped.endswith(lora_weight_keys):
        mapped = mapped[:-len(".weight")]

    return mapped


def _map_llm_tensor(source_key: str, mapped_key: str, tensor: torch.Tensor) -> torch.Tensor:
    del source_key
    lora_suffixes = (
        ".att.w1", ".att.w2",
        ".att.a1", ".att.a2",
        ".att.v1", ".att.v2",
        ".att.g1", ".att.g2",
    )
    if mapped_key.endswith(lora_suffixes):
        return tensor.transpose(0, 1).contiguous()
    return tensor


def _patch_qwen3vl_vision_for_single_image_export(encoder: torch.nn.Module) -> None:
    """Replace HF helper methods that use Tensor.tolist() with traceable single-image variants."""

    def rot_pos_emb_single(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = int(self.spatial_merge_size)
        grid = grid_thw[0]
        height = grid[1]
        width = grid[2]
        device = self.pos_embed.weight.device

        merged_h = torch.div(height, merge_size, rounding_mode="floor")
        merged_w = torch.div(width, merge_size, rounding_mode="floor")
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra = torch.arange(merge_size, device=device)

        row_idx = block_rows[:, None, None, None] * merge_size + intra[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        pos_ids = torch.stack((row_idx, col_idx), dim=-1).to(self.rotary_pos_emb.inv_freq.dtype)
        embeddings = pos_ids.unsqueeze(-1) * self.rotary_pos_emb.inv_freq
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate_single(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = int(self.config.spatial_merge_size)
        grid = grid_thw[0]
        height = grid[1]
        width = grid[2]
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype

        h_range = torch.arange(height, device=device, dtype=dtype)
        w_range = torch.arange(width, device=device, dtype=dtype)
        h_denom = torch.clamp((height - 1).to(dtype), min=1)
        w_denom = torch.clamp((width - 1).to(dtype), min=1)
        h_idxs = h_range * (float(self.num_grid_per_side - 1) / h_denom)
        w_idxs = w_range * (float(self.num_grid_per_side - 1) / w_denom)

        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=self.num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=self.num_grid_per_side - 1)
        dh = h_idxs - h_floor.to(dtype)
        dw = w_idxs - w_floor.to(dtype)

        base_h = h_floor * self.num_grid_per_side
        base_h_ceil = h_ceil * self.num_grid_per_side
        idx00 = (base_h[:, None] + w_floor[None, :]).reshape(-1)
        idx01 = (base_h[:, None] + w_ceil[None, :]).reshape(-1)
        idx10 = (base_h_ceil[:, None] + w_floor[None, :]).reshape(-1)
        idx11 = (base_h_ceil[:, None] + w_ceil[None, :]).reshape(-1)

        w00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1)
        w01 = ((1 - dh)[:, None] * dw[None, :]).reshape(-1)
        w10 = (dh[:, None] * (1 - dw)[None, :]).reshape(-1)
        w11 = (dh[:, None] * dw[None, :]).reshape(-1)

        pos_embed = (
            self.pos_embed(idx00) * w00[:, None]
            + self.pos_embed(idx01) * w01[:, None]
            + self.pos_embed(idx10) * w10[:, None]
            + self.pos_embed(idx11) * w11[:, None]
        )
        pos_embed = (
            pos_embed.reshape(
                torch.div(height, merge_size, rounding_mode="floor"),
                merge_size,
                torch.div(width, merge_size, rounding_mode="floor"),
                merge_size,
                -1,
            )
            .permute(0, 2, 1, 3, 4)
            .flatten(0, 3)
        )
        return pos_embed

    def attention_forward_single(self, hidden_states, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        del cu_seqlens, rotary_pos_emb, kwargs
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        orig_q_dtype = query_states.dtype
        orig_k_dtype = key_states.dtype
        q = query_states.float()
        k = key_states.float()
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        q = ((q * cos) + (torch.cat((-q[..., q.shape[-1] // 2 :], q[..., : q.shape[-1] // 2]), dim=-1) * sin)).to(orig_q_dtype)
        k = ((k * cos) + (torch.cat((-k[..., k.shape[-1] // 2 :], k[..., : k.shape[-1] // 2]), dim=-1) * sin)).to(orig_k_dtype)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = value_states.transpose(0, 1).unsqueeze(0)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    encoder.rot_pos_emb = types.MethodType(rot_pos_emb_single, encoder)
    encoder.fast_pos_embed_interpolate = types.MethodType(fast_pos_embed_interpolate_single, encoder)
    for block in encoder.blocks:
        block.attn.forward = types.MethodType(attention_forward_single, block.attn)


def split_llm(args: argparse.Namespace) -> None:
    model_dir = args.model_dir
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    state: dict[str, torch.Tensor] = {}
    skipped = 0
    for shard in _iter_safetensors(model_dir):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                mapped = _map_llm_key(key)
                if mapped is None:
                    skipped += 1
                    continue
                state[mapped] = _map_llm_tensor(key, mapped, f.get_tensor(key))

    required = ["emb.weight", "head.weight", "ln_out.weight", "ln_out.bias"]
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"Missing required LLM tensors after split: {missing}")

    torch.save(state, out)
    print(f"Wrote {len(state)} LLM tensors to {out} (skipped {skipped} non-LLM tensors)")


class _VisionProjector(torch.nn.Module):
    def __init__(self, model_dir: Path, dtype: torch.dtype):
        super().__init__()
        sys.path.insert(0, str(model_dir))
        try:
            from modeling_modrwkv import VisualAdapter  # type: ignore
            from transformers import Qwen3VLVisionModel
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        finally:
            try:
                sys.path.remove(str(model_dir))
            except ValueError:
                pass

        config = _load_json(model_dir / "config.json")
        vision_config = dict(config["vision_config"])
        vision_config["model_type"] = "qwen3_vl"
        qwen_vision_config = Qwen3VLVisionConfig(**vision_config)
        qwen_vision_config._attn_implementation = "eager"
        qwen_vision_config._attn_implementation_internal = "eager"
        projector_config = config["projector_config"]

        self.encoder = Qwen3VLVisionModel(qwen_vision_config)
        _patch_qwen3vl_vision_for_single_image_export(self.encoder)
        self.proj = VisualAdapter(
            encoder_dim=int(projector_config["encoder_dim"]),
            project_dim=int(projector_config["project_dim"]),
            hidden_dim=projector_config.get("hidden_dim"),
            num_deepstack=int(projector_config.get("num_deepstack") or 0),
            use_conv=bool(config.get("use_conv_in_projector", False)),
        )
        self._load_weights(model_dir)
        self.to(dtype=dtype)
        self.eval()

    def _load_weights(self, model_dir: Path) -> None:
        encoder_state: dict[str, torch.Tensor] = {}
        projector_state: dict[str, torch.Tensor] = {}
        for shard in _iter_safetensors(model_dir):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("model.encoder."):
                        encoder_state[key.removeprefix("model.encoder.")] = f.get_tensor(key)
                    elif key.startswith("model.proj."):
                        projector_state[key.removeprefix("model.proj.")] = f.get_tensor(key)

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Vision encoder weight mismatch: missing={missing}, unexpected={unexpected}")
        missing, unexpected = self.proj.load_state_dict(projector_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Projector weight mismatch: missing={missing}, unexpected={unexpected}")

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        vision_output = self.encoder(pixel_values, image_grid_thw)
        vision_embeds = vision_output.pooler_output
        projected, _ = self.proj(vision_embeds, [])
        return projected.reshape(-1, projected.shape[-1])


class _QwenVisionEncoder(torch.nn.Module):
    def __init__(self, model_dir: Path, dtype: torch.dtype):
        super().__init__()
        sys.path.insert(0, str(model_dir))
        try:
            from transformers import Qwen3VLVisionModel
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
        finally:
            try:
                sys.path.remove(str(model_dir))
            except ValueError:
                pass

        config = _load_json(model_dir / "config.json")
        vision_config = dict(config["vision_config"])
        vision_config["model_type"] = "qwen3_vl"
        qwen_vision_config = Qwen3VLVisionConfig(**vision_config)
        qwen_vision_config._attn_implementation = "eager"
        qwen_vision_config._attn_implementation_internal = "eager"

        self.encoder = Qwen3VLVisionModel(qwen_vision_config)
        _patch_qwen3vl_vision_for_single_image_export(self.encoder)
        self._load_weights(model_dir)
        self.to(dtype=dtype)
        self.eval()

    def _load_weights(self, model_dir: Path) -> None:
        encoder_state: dict[str, torch.Tensor] = {}
        for shard in _iter_safetensors(model_dir):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("model.encoder."):
                        encoder_state[key.removeprefix("model.encoder.")] = f.get_tensor(key)

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Vision encoder weight mismatch: missing={missing}, unexpected={unexpected}")

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        vision_output = self.encoder(pixel_values, image_grid_thw)
        pooler_output = vision_output.pooler_output
        return pooler_output.reshape(-1, pooler_output.shape[-1])


class _QwenVisionAdapter(torch.nn.Module):
    def __init__(self, model_dir: Path, dtype: torch.dtype):
        super().__init__()
        sys.path.insert(0, str(model_dir))
        try:
            from modeling_modrwkv import VisualAdapter  # type: ignore
        finally:
            try:
                sys.path.remove(str(model_dir))
            except ValueError:
                pass

        config = _load_json(model_dir / "config.json")
        projector_config = config["projector_config"]
        self.proj = VisualAdapter(
            encoder_dim=int(projector_config["encoder_dim"]),
            project_dim=int(projector_config["project_dim"]),
            hidden_dim=projector_config.get("hidden_dim"),
            num_deepstack=int(projector_config.get("num_deepstack") or 0),
            use_conv=bool(config.get("use_conv_in_projector", False)),
        )
        self._load_weights(model_dir)
        self.to(dtype=dtype)
        self.eval()

    def _load_weights(self, model_dir: Path) -> None:
        projector_state: dict[str, torch.Tensor] = {}
        for shard in _iter_safetensors(model_dir):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("model.proj."):
                        projector_state[key.removeprefix("model.proj.")] = f.get_tensor(key)

        missing, unexpected = self.proj.load_state_dict(projector_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Projector weight mismatch: missing={missing}, unexpected={unexpected}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        projected, _ = self.proj(input, [])
        return projected.reshape(-1, projected.shape[-1])


def _parse_grid(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--grid-thw must look like 1,24,24")
    t, h, w = parts
    if t <= 0 or h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("--grid-thw values must be positive")
    if h % 2 or w % 2:
        raise argparse.ArgumentTypeError("Qwen3-VL spatial merge size is 2, so grid h/w must be even")
    return t, h, w


def export_vision_onnx(args: argparse.Namespace) -> None:
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    model = _VisionProjector(args.model_dir, dtype=dtype)
    patch_dim = 3 * 2 * 16 * 16
    num_patches = args.grid_thw[0] * args.grid_thw[1] * args.grid_thw[2]
    dummy = torch.zeros(num_patches, patch_dim, dtype=dtype)
    dummy_grid = torch.tensor([args.grid_thw], dtype=torch.int32)

    with torch.inference_mode():
        output = model(dummy, dummy_grid)
    print(
        "Vision/projector shape:",
        f"pixel_values={tuple(dummy.shape)}",
        f"image_grid_thw={tuple(dummy_grid.shape)}",
        f"image_embeddings={tuple(output.shape)}",
    )

    torch.onnx.export(
        model,
        (dummy, dummy_grid),
        out,
        input_names=["pixel_values", "image_grid_thw"],
        output_names=["image_embeddings"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
            "image_embeddings": {0: "num_image_tokens"},
        },
    )
    print(f"Wrote ONNX to {out}")

    if args.mnn_out is not None:
        convert_onnx_to_mnn(out, args.mnn_out, args.mnnconvert)


def export_vision_split_onnx(args: argparse.Namespace) -> None:
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    encoder_out = args.encoder_out
    adapter_out = args.adapter_out
    encoder_out.parent.mkdir(parents=True, exist_ok=True)
    adapter_out.parent.mkdir(parents=True, exist_ok=True)

    patch_dim = 3 * 2 * 16 * 16
    num_patches = args.grid_thw[0] * args.grid_thw[1] * args.grid_thw[2]
    dummy_pixels = torch.zeros(num_patches, patch_dim, dtype=dtype)

    dummy_grid = torch.tensor([args.grid_thw], dtype=torch.int32)

    encoder = _QwenVisionEncoder(args.model_dir, dtype=dtype)
    with torch.inference_mode():
        pooler_output = encoder(dummy_pixels, dummy_grid)
    print(
        "Vision encoder shape:",
        f"pixel_values={tuple(dummy_pixels.shape)}",
        f"image_grid_thw={tuple(dummy_grid.shape)}",
        f"pooler_output={tuple(pooler_output.shape)}",
    )
    torch.onnx.export(
        encoder,
        (dummy_pixels, dummy_grid),
        encoder_out,
        input_names=["pixel_values", "image_grid_thw"],
        output_names=["pooler_output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
            "pooler_output": {0: "num_merged_patches"},
        },
    )
    print(f"Wrote vision encoder ONNX to {encoder_out}")

    adapter = _QwenVisionAdapter(args.model_dir, dtype=dtype)
    dummy_adapter_input = torch.zeros_like(pooler_output)
    with torch.inference_mode():
        image_embeddings = adapter(dummy_adapter_input)
    print(
        "Vision adapter shape:",
        f"input={tuple(dummy_adapter_input.shape)}",
        f"image_embeddings={tuple(image_embeddings.shape)}",
    )
    torch.onnx.export(
        adapter,
        (dummy_adapter_input,),
        adapter_out,
        input_names=["input"],
        output_names=["image_embeddings"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "num_merged_patches"},
            "image_embeddings": {0: "num_image_tokens"},
        },
    )
    print(f"Wrote vision adapter ONNX to {adapter_out}")

    if args.encoder_mnn_out is not None:
        convert_onnx_to_mnn(encoder_out, args.encoder_mnn_out, args.mnnconvert)
    if args.adapter_mnn_out is not None:
        convert_onnx_to_mnn(adapter_out, args.adapter_mnn_out, args.mnnconvert)


def convert_onnx_to_mnn(onnx_path: Path, mnn_path: Path, mnnconvert: Path | None) -> None:
    exe = str(mnnconvert) if mnnconvert is not None else shutil.which("MNNConvert")
    if exe is None:
        raise FileNotFoundError("MNNConvert not found; pass --mnnconvert or put it on PATH")
    mnn_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-f",
        "ONNX",
        "--modelFile",
        str(onnx_path),
        "--MNNModel",
        str(mnn_path),
        "--bizCode",
        "MNN",
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote MNN to {mnn_path}")


def convert_llm_gguf(args: argparse.Namespace) -> None:
    pth = args.pth
    if not pth.exists() or args.force_split:
        split_args = argparse.Namespace(model_dir=args.model_dir, out=pth)
        split_llm(split_args)

    vocab = args.vocab or (args.model_dir / "wr_vocab_v20230424.txt")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "convert_rwkv_pth_to_gguf.py"),
        "--outfile",
        str(args.out),
        "--outtype",
        args.outtype,
        str(pth),
        str(vocab),
    ]
    if args.no_lazy:
        cmd.insert(2, "--no-lazy")
    subprocess.run(cmd, check=True)


def convert_all(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name
    pth = args.out_dir / f"{stem}-rwkv.pth"
    gguf = args.out_dir / f"{stem}-{args.outtype}.gguf"
    encoder_onnx = args.out_dir / f"{stem}-vision-encoder.onnx"
    adapter_onnx = args.out_dir / f"{stem}-vision-adapter.onnx"
    encoder_mnn = args.out_dir / f"{stem}-vision-encoder.mnn" if args.mnn else None
    adapter_mnn = args.out_dir / f"{stem}-vision-adapter.mnn" if args.mnn else None

    split_llm(argparse.Namespace(model_dir=args.model_dir, out=pth))
    if args.gguf:
        convert_llm_gguf(
            argparse.Namespace(
                model_dir=args.model_dir,
                pth=pth,
                out=gguf,
                outtype=args.outtype,
                vocab=args.vocab,
                force_split=False,
                no_lazy=args.no_lazy,
            )
        )
    copy_runtime_metadata(args.model_dir, args.out_dir)
    export_vision_split_onnx(
        argparse.Namespace(
            model_dir=args.model_dir,
            encoder_out=encoder_onnx,
            adapter_out=adapter_onnx,
            grid_thw=args.grid_thw,
            dtype=args.vision_dtype,
            opset=args.opset,
            encoder_mnn_out=encoder_mnn,
            adapter_mnn_out=adapter_mnn,
            mnnconvert=args.mnnconvert,
        )
    )


def copy_runtime_metadata(model_dir: Path, out_dir: Path) -> None:
    for name in [
        "chat_template.jinja",
        "tokenizer_config.json",
        "processor_config.json",
        "generation_config.json",
        "wr_vocab_v20230424.txt",
    ]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    tokens = {
        "vision_start_token": "<|vision_start|>",
        "vision_start_token_id": 65530,
        "vision_end_token": "<|vision_end|>",
        "vision_end_token_id": 65531,
        "image_token": "<|image_pad|>",
        "image_token_id": 65532,
        "vision_encoder_input": "pixel_values",
        "vision_encoder_output": "pooler_output",
        "vision_adapter_input": "input",
        "vision_adapter_output": "image_embeddings",
        "mobile_prefill_order": [
            "text_before_image",
            "vision_start_token",
            "image_embeddings",
            "vision_end_token",
            "text_after_image",
        ],
    }
    with (out_dir / "vl_runtime_tokens.json").open("w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ModRWKV/RWKV-VL HF exports into mobile assets")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    split = subparsers.add_parser("split-llm", help="write a legacy RWKV .pth from the VL safetensors")
    split.add_argument("model_dir", type=Path)
    split.add_argument("--out", type=Path, required=True)
    split.set_defaults(func=split_llm)

    gguf = subparsers.add_parser("llm-gguf", help="split LLM weights and call convert_rwkv_pth_to_gguf.py")
    gguf.add_argument("model_dir", type=Path)
    gguf.add_argument("--pth", type=Path, required=True)
    gguf.add_argument("--out", type=Path, required=True)
    gguf.add_argument("--outtype", choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="f16")
    gguf.add_argument("--vocab", type=Path)
    gguf.add_argument("--force-split", action="store_true")
    gguf.add_argument("--no-lazy", action="store_true")
    gguf.set_defaults(func=convert_llm_gguf)

    vision = subparsers.add_parser("vision-onnx", help="export dynamic Qwen3-VL vision tower + ModRWKV projector as one ONNX")
    vision.add_argument("model_dir", type=Path)
    vision.add_argument("--out", type=Path, required=True)
    vision.add_argument("--grid-thw", type=_parse_grid, default=(1, 24, 24), help="trace/example input grid, e.g. 1,24,24")
    vision.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f16")
    vision.add_argument("--opset", type=int, default=17)
    vision.add_argument("--mnn-out", type=Path)
    vision.add_argument("--mnnconvert", type=Path)
    vision.set_defaults(func=export_vision_onnx)

    vision_split = subparsers.add_parser("vision-split-onnx", help="export dynamic Qwen3-VL vision encoder and adapter as separate ONNX files")
    vision_split.add_argument("model_dir", type=Path)
    vision_split.add_argument("--encoder-out", type=Path, required=True)
    vision_split.add_argument("--adapter-out", type=Path, required=True)
    vision_split.add_argument("--grid-thw", type=_parse_grid, default=(1, 24, 24), help="trace/example input grid, e.g. 1,24,24")
    vision_split.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f16")
    vision_split.add_argument("--opset", type=int, default=17)
    vision_split.add_argument("--encoder-mnn-out", type=Path)
    vision_split.add_argument("--adapter-mnn-out", type=Path)
    vision_split.add_argument("--mnnconvert", type=Path)
    vision_split.set_defaults(func=export_vision_split_onnx)

    all_parser = subparsers.add_parser("all", help="produce RWKV .pth and dynamic vision encoder + adapter ONNX, optionally GGUF/MNN")
    all_parser.add_argument("model_dir", type=Path)
    all_parser.add_argument("--out-dir", type=Path, required=True)
    all_parser.add_argument("--name", default="modrwkv-vl")
    all_parser.add_argument("--outtype", choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"], default="f16")
    all_parser.add_argument("--vocab", type=Path)
    all_parser.add_argument("--grid-thw", type=_parse_grid, default=(1, 24, 24))
    all_parser.add_argument("--vision-dtype", choices=["f32", "f16", "bf16"], default="f16")
    all_parser.add_argument("--opset", type=int, default=17)
    all_parser.add_argument("--gguf", action="store_true", help="also call convert_rwkv_pth_to_gguf.py")
    all_parser.add_argument("--mnn", action="store_true")
    all_parser.add_argument("--mnnconvert", type=Path)
    all_parser.add_argument("--no-lazy", action="store_true")
    all_parser.set_defaults(func=convert_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
