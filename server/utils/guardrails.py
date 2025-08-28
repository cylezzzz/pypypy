def mode_is_creative(settings: dict) -> bool:
    return str(settings.get("mode","creative")).lower() == "creative"
