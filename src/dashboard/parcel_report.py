"""Printable explanations of parcel recommendations for planning review."""

from io import BytesIO
from math import isfinite
from textwrap import fill

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import colormaps

from .parcel_attention import load_parcel_attention

from .planner_explanation import planner_explanation


ACTION_GUIDANCE = {
    "retain_formalise": ("Retain and formalise", "Review tenure and recognition options for the existing settlement."),
    "retain_upgrade": ("Retain and upgrade", "Review service and infrastructure improvements that let households remain in place."),
    "in_situ_upgrade_priority": ("Priority in-situ upgrade", "Prioritise an on-site assessment of gaps in services and housing conditions."),
    "flood_resilient_upgrade": ("Flood-resilient upgrade", "Assess drainage, safe access and building adaptations that could reduce exposure in place."),
    "compensated_resettlement": ("Compensated resettlement", "Assess safe in-place alternatives, verify the hazard independently and consult affected households before considering relocation."),
    "insufficient_evidence": ("Insufficient evidence", "Collect missing flood or building information before selecting a policy action."),
}


def _percent(value):
    return f"{value:.1%}" if value is not None and isfinite(value) else "Unavailable"


INK = "#172d35"
MUTED = "#425b63"
GREEN = "#116857"
PAPER = "#f5f7f6"


def _text(fig, x, y, text, *, size=10, color=INK, weight="normal", width=None):
    if width is not None:
        text = fill(text, width=width)
    fig.text(x, y, text, fontsize=size, color=color, fontweight=weight,
             va="top", parse_math=False, linespacing=1.25)
    return y - (text.count("\n") + 1) * (size + 2) / 841.68


def _box(fig, x, top, width, height, *, color="white", edge=None):
    fig.add_artist(FancyBboxPatch((x, top - height), width, height,
                                  boxstyle="round,pad=0,rounding_size=.012",
                                  transform=fig.transFigure, facecolor=color,
                                  edgecolor=edge or color, linewidth=.7, zorder=-1))


def _page(parcel, page, title, subtitle):
    fig = Figure(figsize=(8.27, 11.69), facecolor=PAPER)
    _box(fig, .046, .959, .908, .139, color="#173c3a")
    _text(fig, .078, .932, "NAIROBI RISK MAPPING", size=9, color="#aee1cc", weight="bold")
    _text(fig, .078, .903, title, size=26, color="white", weight="bold")
    _text(fig, .079, .852, subtitle, size=10, color="#e4f3ed")
    _text(fig, .069, .043, f"Planning screening  |  {parcel['id']}", size=8, color=MUTED)
    _text(fig, .884, .043, f"{page} / 2", size=8, color=MUTED, weight="bold")
    return fig


def _image_panel(fig, attention):
    top = .43
    if attention is None:
        _box(fig, .07, top, .86, .098, color="white", edge="#dbe5e0")
        _text(fig, .091, top - .022, "Parcel imagery unavailable", weight="bold")
        _text(fig, .091, top - .047, "This export needs a compatible satellite raster covering the selected parcel.", size=9, color=MUTED)
        return
    has_heat = attention.heat is not None
    _text(fig, .07, top, "Satellite imagery", size=9, weight="bold")
    _text(fig, .515, top, "Grad-CAM overlay" if has_heat else "Flood model attention", size=9, weight="bold")
    for x, show_heat in ((.07, False), (.515, True)):
        if show_heat and not has_heat:
            _box(fig, x, .405, .415, .27, color="white", edge="#dbe5e0")
            _text(fig, x + .02, .372, attention.note, size=9, color=MUTED, width=44)
            continue
        ax = fig.add_axes([x, .135, .415, .27])
        ax.imshow(attention.rgb, interpolation="nearest")
        if show_heat:
            ax.imshow(attention.heat, cmap="inferno", vmin=0, vmax=1,
                      alpha=.68 * attention.heat, interpolation="bilinear")
        mx, my = attention.marker
        ax.scatter(mx, my, color="white", marker="+", linewidths=1.4, s=62)
        ax.set_axis_off()
    if has_heat:
        for i, level in enumerate((.20, .4, .6, .8, 1.0)):
            fig.add_artist(Rectangle((.515 + i * .022, .117), .020, .008,
                                     transform=fig.transFigure,
                                     facecolor=colormaps["inferno"](level), linewidth=0))
        _text(fig, .637, .126, "Lower to higher relative attention", size=8, color=MUTED)
    note = attention.note if has_heat else "Crosshair: selected parcel."
    size_note = f" Image width: about {attention.image_width_m:.0f} m." if attention.image_width_m else ""
    _text(fig, .07, .105, "Sentinel-2 RGB. " + note + size_note, size=8, color=MUTED, width=110)
    _text(fig, .07, .075, "Grad-CAM explains a new run on the local tile. Verify flood conditions independently.", size=8, color=MUTED)


def _decision_page(parcel, action, guidance, planning, attention):
    place = parcel.get("settlement", "").title() or "Settlement unavailable"
    fig = _page(parcel, 1, "Parcel planning brief", f"{place}  |  Parcel {parcel['id']}")
    _box(fig, .07, .801, .86, .129, color="#e5f1ea")
    _text(fig, .092, .781, "SUGGESTED POLICY ACTION", size=9, color=GREEN, weight="bold")
    _text(fig, .092, .751, action, size=19, weight="bold")
    _text(fig, .092, .707, guidance, size=9, width=106)
    cards = (
        ("Flood model score", _percent(parcel.get("flood_proba"))),
        ("Deprivation estimate", (parcel.get("deprivation_class") or "Unavailable").title()),
        ("Policy confidence", _percent(parcel.get("confidence")) if parcel["action"] != "insufficient_evidence" else "Unavailable"),
    )
    for index, (name, value) in enumerate(cards):
        x = .07 + index * .295
        _box(fig, x, .650, .27, .082, color="white", edge="#dbe5e0")
        _text(fig, x + .018, .633, name, size=8, color=MUTED, weight="bold")
        _text(fig, x + .018, .610, value, size=15, weight="bold")
    _text(fig, .07, .543, "How the recommendation was scored", size=13, color=GREEN, weight="bold")
    bottom = _text(fig, .07, .513, planning["confidence"], size=9, width=104)
    if planning["gate"]:
        _text(fig, .07, bottom - .007, planning["gate"], size=8, color=MUTED, width=112)
    _image_panel(fig, attention)
    return fig


def _evidence_page(parcel, planning, lift_note):
    fig = _page(parcel, 2, "Evidence for field review", "Building indicators, interpretation and verification")
    _text(fig, .07, .794, "How the building indicators were used", size=13, color=GREEN, weight="bold")
    y = _text(fig, .07, .762, planning["model_note"] + " " + planning["deprivation_result"], size=9, width=105) - .014
    y = _text(fig, .07, y, planning["input_note"], size=8, color=MUTED, width=114) - .014
    styles = {
        "favorable": ("#e5f1ea", GREEN),
        "unfavorable": ("#fae8eb", "#961c3b"),
        "neutral": ("#ebeff0", MUTED),
    }
    if not planning["features"]:
        y = _text(fig, .07, y, "No building indicators were saved for this parcel.", size=9) - .014
    for feature in planning["features"]:
        bg, fg = styles[feature["condition"]]
        _box(fig, .07, y, .86, .071, color="white", edge="#dbe5e0")
        _text(fig, .087, y - .012, feature["label"], weight="bold", size=9)
        _box(fig, .646, y - .009, .267, .023, color=bg)
        _text(fig, .657, y - .015, feature["comparison"], size=8, color=fg, weight="bold")
        detail = ". ".join(part for part in (feature["scale"], feature["meaning"]) if part)
        _text(fig, .087, y - .037, detail, size=8, color=MUTED, width=112)
        y -= .078
    y = _text(fig, .07, y - .006, planning["signal_note"], size=8, color=MUTED, width=114) - .010
    if planning["has_model_signals"]:
        strongest = planning["signals"][0]
        y = _text(fig, .07, y, f"{strongest['label']}: {strongest['signal']}", size=8, width=114) - .010
    y = _text(fig, .07, y, "Before acting", size=13, color=GREEN, weight="bold") - .007
    y = _text(fig, .07, y, "Verify flood history, building conditions and service access with local data and residents.", size=9, width=105) - .006
    y = _text(fig, .07, y, planning["confidence_limit"], size=8, color=MUTED, width=114) - .006
    _text(fig, .07, y, lift_note, size=8, color=MUTED, width=114)
    return fig


def build_parcel_pdf(parcel: dict, lift_note: str) -> bytes:
    """Make the decision brief and field review pages for the selected parcel."""
    action, guidance = ACTION_GUIDANCE.get(parcel["action"], ACTION_GUIDANCE["insufficient_evidence"])
    planning = parcel.get("planning") or planner_explanation(parcel)
    attention = load_parcel_attention(parcel)
    with BytesIO() as output:
        with PdfPages(output, metadata={"Title": f"Parcel planning report - {parcel['id']}"}) as report:
            report.savefig(_decision_page(parcel, action, guidance, planning, attention))
            report.savefig(_evidence_page(parcel, planning, lift_note))
        return output.getvalue()
