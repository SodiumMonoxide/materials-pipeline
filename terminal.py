#!/usr/bin/env python3
"""Numbered terminal menu for Materials Pipeline."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import signal
import sys
from unicodedata import east_asian_width

from pipeline import ROOT, resolve_path, run_cli


EXAMPLE = ROOT / "examples" / "smoke.json"
DISCOVERY = ROOT / "examples" / "discovery.json"
CALCULATIONS = {
    "1": ("ferroelectric", "铁电极化"),
    "2": ("optical", "光学性质"),
    "3": ("mechanical", "力学性质"),
    "4": (None, "运行整份清单"),
}
PROPERTY_NAMES = {prop: label for prop, label in CALCULATIONS.values() if prop}


def text_width(text):
    return sum(2 if east_asian_width(char) in "WF" else 1 for char in text)


def pad(text, width):
    return text + " " * max(0, width - text_width(text))


def section(label, width):
    title = f" {label} "
    side = width - text_width(title)
    print("=" * (side // 2) + title + "=" * (side - side // 2))


def menu_row(left, right, width):
    column = width // 2
    if text_width(left) + 2 <= column and text_width(right) <= width - column:
        print(pad(left, column) + right)
    else:
        print(left)
        print(right)


def show_menu():
    width = max(24, min(74, shutil.get_terminal_size((74, 24)).columns))
    border = "+" + "-" * (width - 2) + "+"
    print(f"\n{border}")
    if width >= 48:
        lines = ("   o---o       MATERIALS PIPELINE",
                 "  /|  /|       VASP 材料性质批量计算",
                 " o---o |")
    else:
        lines = (" MATERIALS PIPELINE", " VASP 材料性质批量计算")
    for line in lines:
        print("|" + pad(line, width - 2) + "|")
    print(border)
    section("计算项目", width)
    for first, second in (("1", "2"), ("3", "4")):
        menu_row(f" {first}) {CALCULATIONS[first][1]}", f" {second}) {CALCULATIONS[second][1]}", width)
    section("批次工具", width)
    menu_row("11) 环境检查", "12) 查看批次汇总", width)
    print("13) 检索铁电候选结构")
    print(" 0) 退出\n")


def choose(prompt, choices):
    while True:
        option = input(prompt).strip()
        if option in choices:
            return option
        print("请输入菜单中的编号。")


def ask_path(label, default):
    value = input(f"{label} [{default}]\n> ").strip()
    return resolve_path(Path.cwd(), value) if value else default


def suggested_output(manifest):
    return Path.cwd() / "runs" / manifest.stem


@dataclass
class Selection:
    manifest: Path = EXAMPLE
    output: Path = suggested_output(EXAMPLE)

    def ask_manifest(self, default=None):
        manifest = ask_path("清单路径", default or self.manifest)
        if manifest != self.manifest:
            self.manifest = manifest
            self.output = suggested_output(manifest)
        return manifest

    def ask_output(self):
        self.output = ask_path("输出目录", self.output)
        return self.output


def run_in_menu(args):
    # The driver owns SIGINT/SIGTERM during a batch. Restore menu input behavior on return.
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        status = run_cli(args, prog="materials-pipeline")
    except KeyboardInterrupt:
        status = 130
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print({0: "操作结束。", 1: "部分任务需要处理，请查看批次汇总。",
           2: "请检查上方的配置或环境错误。", 3: "有铁电候选等待选择，请查看候选报告。",
           130: "操作已中断。"}.get(status, f"退出码 {status}"))


def calculation_menu(option, selection):
    prop, label = CALCULATIONS[option]
    print(f"\n{label}")
    if prop == "optical":
        print("若清单要求使用力学平衡结构或铁电极性结构，会一并执行前置步骤。")
    print(" 1) 开始计算   2) 准备输入   0) 返回")
    action = choose("请选择操作 > ", {"0", "1", "2"})
    if action == "0":
        return
    manifest = selection.ask_manifest()
    output = selection.ask_output()
    args = ["run" if action == "1" else "prepare", str(manifest), "--out", str(output)]
    if prop is not None:
        args.extend(("--property", prop))
    run_in_menu(args)


def show_summary(output):
    candidates = [output / f"{action}-summary.json" for action in ("run", "prepare", "discover")]
    files = [path for path in candidates if path.is_file()]
    if not files:
        print(f"目录中还没有批次汇总\n{output}")
        return
    path = max(files, key=lambda candidate: candidate.stat().st_mtime_ns)
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n最近批次汇总\n{path}")
    for row in report["stages"]:
        prop = PROPERTY_NAMES.get(row["property"], row["property"])
        print(f"  {row['material']} / {prop}  [{row['status']}]")
        if message := row.get("message"):
            print(f"    {message}")
        for field in ("result", "convergence_report"):
            if link := row.get(field):
                print(f"    {link}")
    if not report["stages"]:
        print("  还没有已报告的任务。")
    table = path.with_suffix(".csv")
    if table.is_file():
        print(f"CSV  {table}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        return run_cli(argv, prog="materials-pipeline")

    selection = Selection()
    try:
        while True:
            show_menu()
            option = choose("请输入编号 > ", {*CALCULATIONS, "11", "12", "13", "0"})
            if option == "0":
                return 0
            try:
                if option in CALCULATIONS:
                    calculation_menu(option, selection)
                elif option == "11":
                    run_in_menu(["doctor", str(selection.ask_manifest())])
                elif option == "13":
                    default = DISCOVERY if selection.manifest == EXAMPLE else selection.manifest
                    manifest = selection.ask_manifest(default)
                    run_in_menu(["discover", str(manifest), "--out", str(selection.ask_output())])
                else:
                    show_summary(selection.ask_output())
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"无法完成操作：{error}")
    except EOFError:
        print()
        return 0
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
