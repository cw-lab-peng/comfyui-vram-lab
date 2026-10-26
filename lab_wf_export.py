"""
lab_wf_export.py — build ComfyUI workflows from a compact graph spec.

Two output formats:

* **API format** (`build_api`) — the dict you POST to `/prompt`.
  This is what `lab_vram.py` submits.
* **UI format** (`build_ui`) — the `{nodes, links, widgets_values}` JSON that the
  ComfyUI *web* front-end can load by drag & drop.

The UI format is the fragile one: `widgets_values` is a positional array, and the
order differs per node type *and* per ComfyUI version. So this module **asks the
running server** (`/object_info/<node>`) for the real input order instead of
hard-coding it.

Usage
-----
    python lab_wf_export.py                         # dump every graph as UI JSON
    python lab_wf_export.py --url http://127.0.0.1:8189 --out workflows
    python lab_wf_export.py --only qwen             # just the graphs whose name matches

Edit `MODELS` / `GRAPHS` below to match the checkpoint files you have locally.
"""
import argparse
import json
import os
import urllib.request

# ---------------------------------------------------------------- config
POS = "a red apple on a wooden table, soft window light, shallow depth of field, photorealistic"
NEG = "lowres, blurry, watermark, text, logo, signature"

# Filenames of the weights this lab was measured with. Change them to whatever
# you have installed (uname -a for ComfyUI: /models/{diffusion_models,text_encoders,vae})
MODELS = {
    "zimage_unet": "z-image-turbo-fp8-e4m3fn.safetensors",
    "zimage_clip": "qwen_3_4b_fp8_mixed.safetensors",
    "zimage_vae": "ae.safetensors",
    "sdxl_ckpt": "waiIllustriousSDXL_v170.safetensors",
    "gguf_unet": "qwen-image-2.1-UC-Q4_K_M.gguf",
    "gguf_clip": "qwen3vl_8b_w4a8.safetensors",
    "gguf_vae": "qwen_image_2.1_vae_bf16.safetensors",
}

SPEC_CACHE = {}


def _spec(url, node):
    key = (url, node)
    if key not in SPEC_CACHE:
        with urllib.request.urlopen("%s/object_info/%s" % (url, node), timeout=20) as r:
            SPEC_CACHE[key] = json.load(r)[node]
    return SPEC_CACHE[key]


# ---------------------------------------------------------------- graph specs
# A graph is a list of (node_id, class_type, inputs) where an input value is
# either a literal (widget) or ("<upstream_node_id>", "<output_slot>") for a link.
def graph_zimage(width, height, steps=6, cfg=1.0, seed=43):
    """6-step distilled DiT + its own text encoder (ComfyUI >= 0.28)."""
    return [
        (1, "UNETLoader", {"unet_name": MODELS["zimage_unet"], "weight_dtype": "default"}),
        (2, "CLIPLoader", {"clip_name": MODELS["zimage_clip"], "type": "qwen_image"}),
        (3, "VAELoader", {"vae_name": MODELS["zimage_vae"]}),
        (4, "TextEncodeZImageOmni", {"clip": (2, 0), "prompt": POS, "auto_resize_images": True}),
        (5, "TextEncodeZImageOmni", {"clip": (2, 0), "prompt": NEG, "auto_resize_images": True}),
        (6, "EmptyLatentImage", {"width": width, "height": height, "batch_size": 1}),
        (7, "KSampler", {"model": (1, 0), "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "normal",
                         "positive": (4, 0), "negative": (5, 0),
                         "latent_image": (6, 0), "denoise": 1.0}),
        (8, "VAEDecode", {"samples": (7, 0), "vae": (3, 0)}),
        (9, "SaveImage", {"images": (8, 0), "filename_prefix": "lab_zimage_%dx%d" % (width, height)}),
    ]


def graph_sdxl(width, height, steps=30, cfg=6.0, seed=43):
    """Classic 30-step checkpoint route, vertical latent inside the ckpt."""
    return [
        (1, "CheckpointLoaderSimple", {"ckpt_name": MODELS["sdxl_ckpt"]}),
        (2, "CLIPTextEncode", {"text": POS, "clip": (1, 1)}),
        (3, "CLIPTextEncode", {"text": NEG, "clip": (1, 1)}),
        (4, "EmptyLatentImage", {"width": width, "height": height, "batch_size": 1}),
        (5, "KSampler", {"model": (1, 0), "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler_ancestral", "scheduler": "normal",
                         "positive": (2, 0), "negative": (3, 0),
                         "latent_image": (4, 0), "denoise": 1.0}),
        (6, "VAEDecode", {"samples": (5, 0), "vae": (1, 2)}),
        (7, "SaveImage", {"images": (6, 0), "filename_prefix": "lab_sdxl_%dx%d" % (width, height)}),
    ]


def graph_qwen_gguf(size=1024, steps=25, cfg=1.0, seed=43):
    """Qwen-Image-2.1 GGUF route — needs ComfyUI >= 0.37 (node `TextEncodeQwenImage21`)."""
    return [
        (1, "UnetLoaderGGUF", {"unet_name": MODELS["gguf_unet"]}),
        (2, "CLIPLoader", {"clip_name": MODELS["gguf_clip"], "type": "qwen_image"}),
        (3, "VAELoader", {"vae_name": MODELS["gguf_vae"]}),
        (4, "TextEncodeQwenImage21", {"clip": (2, 0), "prompt": POS, "negative_prompt": NEG,
                                      "resolution": size, "vae": (3, 0)}),
        (5, "KSampler", {"model": (1, 0), "positive": (4, 0), "negative": (4, 1),
                         "latent_image": (4, 2), "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}),
        (6, "VAEDecode", {"samples": (5, 0), "vae": (3, 0)}),
        (7, "SaveImage", {"images": (6, 0), "filename_prefix": "lab_qwen_%d" % size}),
    ]


GRAPHS = [
    ("01-zimage-512x768", graph_zimage(512, 768)),
    ("02-zimage-896x1152", graph_zimage(896, 1152)),
    ("03-zimage-1152x1536", graph_zimage(1152, 1536)),
    ("04-sdxl-1152x1536-30step", graph_sdxl(1152, 1536)),
    ("05-sdxl-896x1152-30step", graph_sdxl(896, 1152)),
    ("06-qwen-gguf-q4-1024", graph_qwen_gguf(1024)),
]


# ---------------------------------------------------------------- builders
def build_api(graph):
    """Graph spec -> the API-format prompt dict accepted by `POST /prompt`."""
    api = {}
    for nid, ntype, inputs in graph:
        node_inputs = {}
        for k, v in inputs.items():
            node_inputs[k] = list(v) if isinstance(v, tuple) else v
        api[str(nid)] = {"class_type": ntype, "inputs": node_inputs}
    return api


def _widget_slots(url, node):
    """Return [(name, kind, type, extra)] in the order the front-end expects.

    kind == 'widget' -> goes into widgets_values (positional!)
    kind == 'link'   -> goes into inputs[] as a connection
    """
    d = _spec(url, node)
    out = []
    for group in ("required", "optional"):
        for name, v in d["input"].get(group, {}).items():
            t = v[0] if isinstance(v, list) and v else v
            extra = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
            if isinstance(t, list):                      # combo box
                out.append((name, "widget", "COMBO", extra))
            elif t in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                out.append((name, "widget", t, extra))
            else:                                        # socket
                out.append((name, "link", t, extra))
    return out


def build_ui(graph, url):
    """Graph spec -> UI-format workflow dict (drag & drop into the web canvas)."""
    nodes, links, out_links = [], [], {}
    link_id = 0
    for order, (nid, ntype, inputs) in enumerate(graph):
        widgets, ins, slot = [], [], 0
        for name, kind, typ, extra in _widget_slots(url, ntype):
            if kind == "link":
                v = inputs.get(name)
                if isinstance(v, tuple):
                    src, sslot = v
                    link_id += 1
                    links.append([link_id, src, sslot, nid, slot, typ])
                    ins.append({"name": name, "type": typ, "link": link_id})
                    out_links.setdefault((src, sslot), []).append(link_id)
                else:
                    ins.append({"name": name, "type": typ, "link": None})
                slot += 1
                continue
            v = inputs.get(name)
            if v is None:                                # fill a sane default
                opts = extra.get("options") or extra.get("values") or []
                if typ == "COMBO":
                    v = opts[0] if opts else "default"
                elif typ == "INT":
                    v = extra.get("default", 0)
                elif typ == "FLOAT":
                    v = extra.get("default", 0.0)
                elif typ == "BOOLEAN":
                    v = extra.get("default", False)
                else:
                    v = ""
            widgets.append(v)
            if extra.get("control_after_generate"):
                widgets.append("fixed")                  # pin the seed
        d = _spec(url, ntype)
        outs = [{"name": t, "type": t, "links": out_links.get((nid, i))}
                for i, t in enumerate(d.get("output") or [])]
        nodes.append({"id": nid, "type": ntype,
                      "pos": [40 + (nid - 1) * 340, 80 + (nid % 3) * 220],
                      "size": [320, 180], "flags": {}, "order": order, "mode": 0,
                      "inputs": ins, "outputs": outs,
                      "properties": {"Node name for S&R": ntype},
                      "widgets_values": widgets})
    return {"last_node_id": max(n[0] for n in graph), "last_link_id": link_id,
            "nodes": nodes, "links": links, "groups": [], "config": {},
            "extra": {"ds": {"scale": 0.65, "offset": [0, 0]}}, "version": 0.4}


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Export ComfyUI workflows (UI + API format)")
    ap.add_argument("--url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    ap.add_argument("--out", default="workflows")
    ap.add_argument("--only", default="", help="substring filter on graph names")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("server:", args.url)
    for name, graph in GRAPHS:
        if args.only and args.only not in name:
            continue
        wf = build_ui(graph, args.url)
        path = os.path.join(args.out, name + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, ensure_ascii=False, indent=1)
        print("  %-28s %d nodes, %d links -> %s" % (name, len(wf["nodes"]), len(wf["links"]), path))


if __name__ == "__main__":
    main()
