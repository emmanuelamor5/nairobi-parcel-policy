"""Shared explanations for the dashboard and its planning PDF."""

from math import isclose, isfinite


FEATURES = {
    "area_m2": ("Building footprint", "Larger", "Smaller", "The area covered by the building. It gives the model a clue about building size."),
    "dist_to_water_m": ("Distance to water", "Farther", "Closer", "Distance to the nearest mapped water point. Longer distances can signal harder access."),
    "dist_to_sanitation_m": ("Distance to sanitation", "Farther", "Closer", "Distance to the nearest mapped sanitation facility. Longer distances can signal harder access."),
    "dist_to_road_m": ("Distance to a road", "Farther", "Closer", "Distance to the nearest mapped road. Longer distances can signal harder access for service and emergency vehicles."),
    "roof_material_score": ("Roof material", "Higher score", "Lower score", "A score for potential roof vulnerability based on mapped materials. Missing or unfamiliar materials can receive a neutral score."),
}

ACTION_INPUTS = {
    "retain_formalise": (False, ("low",)),
    "retain_upgrade": (False, ("medium",)),
    "in_situ_upgrade_priority": (False, ("high",)),
    "flood_resilient_upgrade": (True, ("low", "medium")),
    "compensated_resettlement": (True, ("high",)),
}

INPUT_NOTE = "Scores compare each building with the average of its settlement graph. One standard deviation is one unit of variation across buildings. The saved inputs do not include the original measurements in metres or square metres."
MODEL_NOTE = "GraphSAGE uses the five building indicators and nearby buildings linked in the graph to estimate the likelihood of low, medium and high structural deprivation."
CONFIDENCE_LIMIT = "The score measures support for the selected action within the policy rules. It does not tell you how likely that action is to succeed."


def _probability(value):
    return value is not None and isfinite(value) and 0 <= value <= 1


def _percent(value):
    return f"{value:.1%}"


def _explain_feature(feature):
    name = feature.get("name")
    label, higher, lower, meaning = FEATURES.get(
        name, (feature["label"], "Higher", "Lower", "A building indicator used by the deprivation model."),
    )
    value = feature.get("value")
    condition = "neutral"
    if value is None or not isfinite(value):
        comparison, scale = "Value unavailable", ""
    elif abs(value) < .005:
        comparison, scale = "At the graph average", "0.00 standard deviations from average"
    else:
        comparison = (higher if value > 0 else lower) + " than average"
        direction = "above" if value > 0 else "below"
        scale = f"{abs(value):.2f} standard deviations {direction} the average"
        if name in FEATURES:
            favorable = value > 0 if name == "area_m2" else value < 0
            condition = "favorable" if favorable else "unfavorable"
    weight = feature.get("attribution")
    if weight is not None and not isfinite(weight):
        weight = None
    return {
        "name": name or feature["label"], "label": label, "comparison": comparison,
        "scale": scale, "meaning": meaning, "weight": weight, "condition": condition,
        "is_model": bool(feature.get("attribution_is_model")),
    }


def _explain_confidence(parcel, *, reference_zone=False, override=False, majority_gate=False):
    confidence = parcel.get("confidence")
    rule = ACTION_INPUTS.get(parcel.get("action"))
    if not rule or not _probability(confidence):
        return "Policy confidence is unavailable for this parcel.", ""
    high_hazard, classes = rule
    scores = {item["name"]: item["value"] for item in parcel.get("class_probabilities") or []}
    flood = parcel.get("flood_proba")
    heading = f"Policy confidence: {_percent(confidence)}."
    breakdown = "The policy matrix combines the flood estimate with the deprivation probabilities. The saved inputs do not support a parcel-specific breakdown."
    if all(_probability(scores.get(name)) for name in classes):
        class_total = sum(scores[name] for name in classes)
        factor = (flood if high_hazard else 1 - flood) if _probability(flood) else None
        zone_factor = float(high_hazard)
        use_zone = override or (reference_zone and (factor is None or not isclose(factor * class_total, confidence, abs_tol=1e-6))
                                and isclose(zone_factor * class_total, confidence, abs_tol=1e-6))
        if use_zone:
            factor = zone_factor
        if factor is not None and isclose(factor * class_total, confidence, abs_tol=1e-6):
            hazard = "high" if high_hazard else "low"
            deprivation = " plus ".join(classes)
            breakdown = (f"The selected rule combines {_percent(factor)} support for {hazard} flood hazard "
                         f"with {_percent(class_total)} for {deprivation} deprivation. Multiplying these gives {_percent(confidence)} (rounded).")
            if use_zone:
                breakdown += " The reference-zone rule counts high flood hazard as 100% in this calculation."
    gate = ("The policy rules selected an in-place upgrade because high deprivation did not get more than half the model's support."
            if majority_gate else "")
    return f"{heading} {breakdown}", gate


def planner_explanation(parcel, **policy_flags):
    features = [_explain_feature(feature) for feature in parcel.get("features", [])]
    stored = [item for item in features if item["is_model"] and item["weight"] is not None]
    signals = stored or [item for item in features if item["weight"] not in (0, None)]
    for item in signals:
        weight = item["weight"]
        if stored:
            direction = "supported" if weight > 0 else "weighed against"
            item["signal"] = (f"{direction.capitalize()} the predicted class by {abs(weight):.2f} model score points."
                              if weight else "No change in the predicted class score.")
        else:
            item["signal"] = "Worth checking in the field based on how this input differs from the reference values."
    signal_note = (
        "These model score changes show how each input affected the predicted deprivation class. They are on the model's internal scale, before conversion to probabilities."
        if stored else "Screening indicators flag unusual inputs for review. Recorded per-feature model explanations are unavailable for this parcel."
    )
    confidence, gate = _explain_confidence(parcel, **policy_flags)
    deprivation = parcel.get("deprivation_class")
    probability = parcel.get("class_probability")
    deprivation_result = f"The model selected {deprivation} structural deprivation" if deprivation else "No structural deprivation class is available"
    if deprivation and _probability(probability):
        deprivation_result += f" with {_percent(probability)} support across the three classes"
    return {
        "features": features, "input_note": INPUT_NOTE, "model_note": MODEL_NOTE,
        "signals": sorted(signals, key=lambda item: -abs(item["weight"])),
        "signal_note": signal_note, "has_model_signals": bool(stored),
        "deprivation_result": deprivation_result + ".", "confidence": confidence,
        "confidence_limit": CONFIDENCE_LIMIT, "gate": gate,
    }
