from __future__ import annotations
from typing import Any
import subprocess, json, shutil

def _ollama_available():
    return shutil.which("ollama") is not None

def _ollama_classify(task: str, prompt: str) -> dict|None:
    try:
        if not _ollama_available(): return None
        # very short classification prompt
        q = f"Classify generation task. Input: task={task}, prompt={prompt}. Return JSON with keys: topic (portrait, landscape, character, abstract, video, lipsync), nsfw (true/false)."
        proc = subprocess.run(["ollama","run","llama3"], input=q.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8)
        out = proc.stdout.decode("utf-8","ignore").strip()
        j = None
        for line in out.splitlines()[::-1]:
            line=line.strip()
            if line.startswith("{") and line.endswith("}"):
                j = json.loads(line); break
        return j
    except Exception:
        return None

def select_plan(model_pref: str|None, task: str, prompt: str, inputs: dict[str,Any], quality: str, catalog: dict):
    models = catalog.get("models", [])
    # Determine required tag for task
    tag = {
        "txt2img":"txt2img","img2img":"img2img","inpaint":"inpaint",
        "pose_transfer":"pose","txt2video":"txt2video","img2video":"img2video",
        "lipsync":"lipsync","talking":"talking"
    }.get(task, task)

    compatible = [m for m in models if tag in (m.get("tags") or [])]
    # Group preference: image vs video
    if task in ("txt2img","img2img","inpaint","pose_transfer"):
        compatible = [m for m in compatible if m.get("group")=="image"] or compatible
    else:
        compatible = [m for m in compatible if m.get("group")=="video"] or compatible

    if model_pref:
        preferred = next((m for m in models if m.get("name")==model_pref), None)
        if preferred:
            compatible = [preferred] + [m for m in compatible if m is not preferred]

    ai = _ollama_classify(task, prompt) or {}
    chosen = compatible[0]["name"] if compatible else "AUTO"

    plan = {
        "task": task,
        "quality": quality,
        "model": chosen,
        "models_suggested": [m["name"] for m in compatible[:6]],
        "ai": ai,
        "stages": route_for_task(task, inputs)
    }
    return plan

def route_for_task(task: str, inputs: dict):
    if task == "inpaint":
        return ["mask_detect_or_use","inpaint","refine","export"]
    if task == "pose_transfer":
        return ["pose_detect","control_guides","img2img","refine","export"]
    if task in ("txt2video","img2video"):
        return ["motion_setup","generate_clip","encode","export"]
    if task in ("lipsync","talking"):
        return ["face_align","sync_or_animate","encode","export"]
    return ["generate","refine","export"]
