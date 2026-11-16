"""Write portable reports and a standalone SVG for optical convergence runs."""

from __future__ import annotations

import csv
import io
import json
import math
from bisect import bisect_left, bisect_right
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from .convergence_metrics import Spectrum


STEP_FIELDS = (
    "axis", "previous_attempt", "attempt", "previous_kmesh", "kmesh",
    "previous_nbands", "nbands", "passed", "worst_channel", "worst_ratio",
    "run_status", "coverage_eV", "coverage_ok", "detail", "message",
)
COLORS = ("#18619c", "#117e82", "#b65826", "#7053a2", "#35744e", "#aa4464", "#4c638d")
STATUS = {
    "converged": "已达到设定阈值",
    "not_converged": "尚未达到设定阈值",
    "exhausted": "已用完加密次数",
    "failed": "计算失败",
    "running": "计算中",
    "interrupted": "计算中断",
    "skipped": "已跳过",
}


def mesh_label(mesh):
    return " × ".join(map(str, mesh)) if mesh else ""


def metric_label(value):
    return f"{value:.4g}" if value is not None else "—"


def _write_atomic(path: Path, text: str) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp") as target:
        temporary = Path(target.name)
        try:
            target.write(text)
        except BaseException:
            target.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _csv_report(steps):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, STEP_FIELDS)
    writer.writeheader()
    for step in steps:
        row = {key: step.get(key) for key in STEP_FIELDS}
        for key in ("previous_kmesh", "kmesh"):
            row[key] = "x".join(map(str, row[key])) if row[key] else ""
        writer.writerow(row)
    return stream.getvalue()


def _markdown_report(report):
    low, high = report["window_eV"]
    lines = [
        f'# {report["material_id"]} 光谱收敛报告', "",
        f'**{STATUS.get(report["status"], report["status"])}**', "",
        f'能区为 **{low:g}–{high:g} eV**。相对阈值为 **{report["rtol"]:.3%}**，'
        f'绝对 RMS 阈值为 **{report["atol"]:g}**。', "",
        '每项对比在该能区内检查 ε₁ 与 ε₂ 的全部六个对称张量分量。'
        '每个分量的允许误差为 `atol + rtol × 参考信号 RMS`。计算 ε₁ 对角分量的参考信号时减去真空值 1。'
        '最大误差比为各分量 `差值 RMS / 允许误差` 的最大值。', "",
    ]
    if report.get("message"):
        lines.extend((str(report["message"]), ""))
    if fixed := report.get("fixed_settings"):
        labels = {"encut": "ENCUT", "cshift": "CSHIFT", "sigma": "SIGMA",
                  "omegamax": "OMEGAMAX", "nedos": "NEDOS"}
        values = [f"`{label}={fixed[key]:g}`" for key, label in labels.items() if key in fixed]
        if values:
            lines.extend(("各步使用相同的光谱参数 " + "、".join(values) + "。", ""))
    if report.get("baseline_attempt"):
        lines.append(f'- 初筛尝试目录 `{report["baseline_attempt"]}`')
    if report.get("final_attempt"):
        lines.append(f'- 最后评估的尝试目录 `{report["final_attempt"]}`，'
                     f'k 网格 **{mesh_label(report["final_kmesh"])}**，总带数 **{report["final_nbands"]}**')
    if report.get("kmesh_required") is False:
        lines.append("- 该体系沿三个方向均采用 Γ 点，收敛判据只检查能带数。")
    if report.get("baseline_attempt") or report.get("final_attempt"):
        lines.append("")
    if report["steps"]:
        lines.extend((
            "| 加密项 | k 网格 | 总带数 | 最敏感分量 | 最大误差比 | 判断 |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ))
        for step in report["steps"]:
            axis = {"kmesh": "k 网格", "nbands": "能带"}.get(step["axis"], step["axis"])
            old_mesh, new_mesh = mesh_label(step.get("previous_kmesh")), mesh_label(step.get("kmesh"))
            old_bands, new_bands = step.get("previous_nbands"), step.get("nbands")
            mesh = f"{old_mesh} → {new_mesh}" if old_mesh != new_mesh else new_mesh
            bands = f"{old_bands} → {new_bands}" if old_bands != new_bands else str(new_bands or "—")
            channel = step.get("worst_channel") or "—"
            ratio = metric_label(step.get("worst_ratio"))
            tested_channel = step.get("channels", {}).get(channel, {})
            if tested_channel.get("threshold") == 0 and tested_channel.get("difference_rms", 0) > 0:
                ratio = "∞"
            passed = step.get("passed")
            verdict = "通过" if passed is True else "未通过" if passed is False else "未比较"
            if step.get("coverage_ok") is False:
                verdict += f"（空带覆盖 {metric_label(step.get('coverage_eV'))} eV）"
            lines.append(f"| {axis} | {mesh or '—'} | {bands} | {channel} | {ratio} | {verdict} |")
        lines.append("")
    lines.extend((
        "结论适用于本次结构、响应模型、截断能、光谱展宽及截取的能区。"
        "叠图显示介电张量三个对角分量的平均值。", "",
        "[光谱叠图](convergence.svg) · [逐步记录 CSV](convergence.csv) · [逐分量误差 JSON](convergence.json)", "",
    ))
    return "\n".join(lines)


def _average_trace(spectrum, part):
    return [sum(sample) / 3 for sample in zip(*(spectrum.channels[f"{part}_{axis}"]
                                              for axis in ("xx", "yy", "zz")))]


def _in_window(energy, values, window):
    lo = max(window[0], energy[0])
    hi = min(window[1], energy[-1])
    if lo >= hi:
        return []

    def boundary(x):
        index = bisect_left(energy, x)
        if energy[index] == x:
            return values[index]
        fraction = (x - energy[index - 1]) / (energy[index] - energy[index - 1])
        return values[index - 1] + fraction * (values[index] - values[index - 1])

    start, stop = bisect_right(energy, lo), bisect_left(energy, hi)
    return [(lo, boundary(lo)), *zip(energy[start:stop], values[start:stop]), (hi, boundary(hi))]


def _ticks(lo, hi, divisions):
    step_guess = (hi - lo) / divisions
    power = 10 ** math.floor(math.log10(step_guess))
    step = next(factor * power for factor in (1, 2, 2.5, 5, 10) if factor * power >= step_guess)
    start = math.ceil(lo / step - 1e-12)
    stop = math.floor(hi / step + 1e-12)
    return tuple(index * step for index in range(start, stop + 1))


def _style(index):
    if index == 0:
        return "#697687", "6 3", 1.7
    color = COLORS[(index - 1) % len(COLORS)]
    dash = ("", "5 3", "2 2", "9 4 2 4")[(index - 1) // len(COLORS) % 4]
    return color, dash, 1.45


def _text(x, y, value, attributes=""):
    return f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{escape(str(value))}</text>'


def _panel(curves, window, part, x, y, width, height):
    lines = [_in_window(spectrum.energy, _average_trace(spectrum, part), window)
             for _, spectrum in curves]
    values = [value for line in lines for _, value in line]
    title = "Real part · ε₁" if part == "eps1" else "Imaginary part · ε₂"
    panel = [_text(x, y - 14, title, 'class="panel-title"')]
    if not values:
        panel.append(_text(x, y + 42, "No spectrum in this energy window", 'class="secondary"'))
        return panel

    y_low, y_high = min(values), max(values)
    padding = (y_high - y_low) * 0.09 if y_high > y_low else max(abs(y_low) * 0.1, 0.1)
    y_low, y_high = y_low - padding, y_high + padding
    left, right = window
    px = lambda energy: x + (energy - left) / (right - left) * width
    py = lambda value: y + (y_high - value) / (y_high - y_low) * height

    panel.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="#fbfcfe"/>')
    for tick in _ticks(y_low, y_high, 5):
        yy = py(tick)
        panel.append(f'<path d="M{x} {yy:.2f}h{width}" class="grid"/>')
        panel.append(_text(x - 10, yy + 4, f"{tick:.4g}", 'class="tick" text-anchor="end"'))
    for tick in _ticks(left, right, 6):
        xx = px(tick)
        panel.append(f'<path d="M{xx:.2f} {y + height}v5" class="axis"/>')
        panel.append(_text(xx, y + height + 22, f"{tick:.4g}", 'class="tick" text-anchor="middle"'))
    panel.append(f'<path d="M{x} {y}v{height}h{width}" class="axis"/>')
    for index, line in enumerate(lines):
        if not line:
            continue
        color, dash, stroke = _style(index)
        coordinates = " ".join(f"{px(energy):.2f},{py(value):.2f}" for energy, value in line)
        panel.append(f'<polyline points="{coordinates}" stroke="{color}" stroke-width="{stroke}"'
                     f' stroke-dasharray="{dash or "none"}" fill="none"/>')
    panel.append(_text(x + width / 2, y + height + 45, "Photon energy (eV)", 'text-anchor="middle"'))
    return panel


def _svg_report(report, curves):
    width = 1060
    window = report["window_eV"]
    label_width = min(width - 72, max(225, max((len(label) for label, _ in curves), default=0) * 7 + 45))
    columns = (width - 96) // label_width or 1
    legend_rows = math.ceil(len(curves) / columns)
    height = 478 + 24 * legend_rows if curves else 188
    title = f'{report["material_id"]} · dielectric spectrum overlay'
    details = f"{window[0]:g}–{window[1]:g} eV · trace / 3 shown · criterion uses each tensor component"
    markup = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"'
        f' viewBox="0 0 {width} {height}" role="img" aria-labelledby="plot-title plot-desc">',
        f'<title id="plot-title">{escape(title)}</title>',
        f'<desc id="plot-desc">{escape(details)}</desc>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#243346;font-size:13px}'
        '.title{font-size:25px;font-weight:600}.panel-title{font-size:16px;font-weight:600}'
        '.secondary,.tick{fill:#536578}.tick{font-size:12px}'
        '.grid{stroke:#dfe6ed;stroke-width:1;fill:none}.axis{stroke:#8494a3;stroke-width:1;fill:none}'
        '</style>',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        _text(42, 44, title, 'class="title"'),
        _text(42, 70, details, 'class="secondary"'),
    ]
    if curves:
        for part, x in (("eps1", 72), ("eps2", 590)):
            markup.extend(_panel(curves, window, part, x, 126, 432, 272))
        for index, (label, _) in enumerate(curves):
            row, column = divmod(index, columns)
            x, y = 60 + column * label_width, 472 + row * 24
            color, dash, stroke = _style(index)
            markup.append(f'<path d="M{x} {y-4}h26" stroke="{color}" stroke-width="{stroke}"'
                          f' stroke-dasharray="{dash or "none"}"/>')
            markup.append(_text(x + 34, y, label))
    else:
        markup.append(_text(42, 125, "No spectra are available for the selected material.", 'class="secondary"'))
    return "\n".join([*markup, "</svg>", ""])


def write_reports(directory: Path, report: dict, curves: list[tuple[str, Spectrum]]) -> None:
    """Publish the spectrum overlay, human-readable report, and structured diagnostics."""
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "convergence.csv": _csv_report(report["steps"]),
        "convergence.md": _markdown_report(report),
        "convergence.svg": _svg_report(report, curves),
        "convergence.json": json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    }
    for name, text in outputs.items():
        _write_atomic(directory / name, text)
