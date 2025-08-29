from __future__ import annotations
from typing import Any, Dict, List, Optional
import subprocess
import json
import shutil


def _ollama_available() -> bool:
    """Check if `ollama` binary is available on PATH."""
    return shutil.which("ollama") is not None


def _ollama_classify(task: str, prompt: str) -> Optional[Dict[str, Any]]:
    """
    Ask a small local LLM via Ollama to classify the generation task.
    Returns a dict like {"topic": "...", "nsfw": true/false} or None on failure.
    """
    if not _ollama_available():
        return None
    try:
        # Very short classification prompt to keep latency low.
        q = (
            "Classify generation task. "
            f"Input: task={task}, prompt={prompt}. "
            "Return JSON with keys: topic (portrait, landscape, character, abstract, video, lipsync), "
            "nsfw (true/false)."
        )
        proc = subprocess.run(
            ["ollama", "run", "llama3"],
            input=q.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=8,
            check=False,
        )
        out = (proc.stdout or b"").decode("utf-8", "ignore").strip()
        # Try to find a JSON object in the output (scan from bottom up)
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    j = json.loads(line)
                    if isinstance(j, dict):
                        return j
                except json.JSONDecodeError:
                    continue
        return None
    except Exception:
        return None


def _task_to_tag(task: str) -> str:
    """Map high-level task to required model tag."""
    return {
        "txt2img": "txt2img",
        "img2img": "img2img",
        "inpaint": "inpaint",
        "pose_transfer": "pose",
        "txt2video": "txt2video",
        "img2video": "img2video",
        "lipsync": "lipsync",
        "talking": "talking",
    }.get(task, task)


def _filter_by_group(task: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prefer 'image' group for image tasks, 'video' for video/talking tasks.
    Fallback to original list if no matches in preferred group exist.
    """
    if task in ("txt2img", "img2img", "inpaint", "pose_transfer"):
        img = [m for m in candidates if m.get("group") == "image"]
        return img or candidates
    else:
        vid = [m for m in candidates if m.get("group") == "video"]
        return vid or candidates


def _is_model_compatible(model: Dict[str, Any], tag: str, group_hint: Optional[str]) -> bool:
    tags = model.get("tags") or []
    if tag and tag not in tags:
        return False
    if group_hint:
        return model.get("group") == group_hint
    return True


def route_for_task(task: str, inputs: Dict[str, Any]) -> List[str]:
    """
    Return a list of pipeline stages for the given task.
    (The 'inputs' param is kept for future branching, if needed.)
    """
    if task == "inpaint":
        return ["mask_detect_or_use", "inpaint", "refine", "export"]
    if task == "pose_transfer":
        return ["pose_detect", "control_guides", "img2img", "refine", "export"]
    if task in ("txt2video", "img2video"):
        return ["motion_setup", "generate_clip", "encode", "export"]
    if task in ("lipsync", "talking"):
        return ["face_align", "sync_or_animate", "encode", "export"]
    return ["generate", "refine", "export"]


def select_plan(
    model_pref: Optional[str],
    task: str,
    prompt: str,
    inputs: Dict[str, Any],
    quality: str,
    catalog: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a generation plan:
      - Resolve compatible models by required tag + group preference
      - Respect model_pref only if it remains compatible
      - Optionally enrich with a tiny Ollama-based classification
    """
    models: List[Dict[str, Any]] = catalog.get("models", []) or []
    req_tag = _task_to_tag(task)

    # 1) filter by required tag
    tagged = [m for m in models if req_tag in (m.get("tags") or [])]

    # 2) apply group preference (image/video)
    compatible = _filter_by_group(task, tagged)

    # 3) handle preferred model – only use if it is compatible
    if model_pref:
        preferred = next((m for m in models if m.get("name") == model_pref), None)
        if preferred is not None:
            # Determine the target group hint used above
            group_hint = "image" if task in ("txt2img", "img2img", "inpaint", "pose_transfer") else "video"
            if _is_model_compatible(preferred, req_tag, group_hint):
                # Put preferred at the front, deduplicate
                compatible = [preferred] + [m for m in compatible if m.get("name") != preferred.get("name")]
            # else: ignore incompatible preference (do not break the pipeline)

    # 4) choose primary model (or AUTO if none)
    chosen = compatible[0]["name"] if compatible and "name" in compatible[0] else "AUTO"

    # 5) try a lightweight task classification (optional)
    ai = _ollama_classify(task, prompt) or {}

    plan: Dict[str, Any] = {
        "task": task,
        "quality": quality,
        "model": chosen,
        "models_suggested": [m.get("name", "unknown") for m in compatible[:6]],
        "ai": ai,
        "stages": route_for_task(task, inputs),
    }
    return plan
