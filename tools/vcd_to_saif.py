#!/usr/bin/env python3
"""Convert the scalar/vector activity in a VCD into a compact SAIF file.

This intentionally small converter is sufficient for the representative RTL
workload produced by ``make activity``.  It keeps the VCD hierarchy, records
zero/one/unknown dwell time and transition count for every declared net, and
does not require a commercial waveform tool.  The resulting SAIF is an
activity artifact for an OpenLane/OpenROAD power run; it is not itself a power
measurement.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Net:
    path: tuple[str, ...]
    name: str
    width: int
    identifier: str
    value: str = "x"
    last_time: int = 0
    t0: int = 0
    t1: int = 0
    tx: int = 0
    toggles: int = 0

    @property
    def saif_path(self) -> str:
        return "/".join((*self.path, self.name))

    def add_interval(self, value: str, duration: int) -> None:
        if duration <= 0:
            return
        normal = normalise_value(value)
        if normal == "0":
            self.t0 += duration
        elif normal == "1":
            self.t1 += duration
        else:
            self.tx += duration

    def change(self, value: str, timestamp: int) -> None:
        self.add_interval(self.value, timestamp - self.last_time)
        old = normalise_value(self.value)
        new = normalise_value(value)
        if old != new:
            self.toggles += 1
        self.value = value
        self.last_time = timestamp


def normalise_value(value: str) -> str:
    value = value.lower()
    if value and all(bit == "0" for bit in value):
        return "0"
    if value and all(bit == "1" for bit in value):
        return "1"
    return "x"


def parse_timescale(line: str) -> str:
    match = re.search(r"\$timescale\s+([^\s]+)\s*([^\s]*)", line, re.I)
    if not match:
        return "1 ns"
    return f"{match.group(1)} {match.group(2)}".strip()


def parse_vcd(path: Path) -> tuple[list[Net], int, str, str]:
    nets: list[Net] = []
    # Some simulators reuse a short VCD identifier in separate scopes.  Keep
    # every declaration so activity is mirrored into each hierarchical net.
    by_id: dict[str, list[Net]] = {}
    scope: list[str] = []
    timescale = "1 ns"
    end_time = 0
    in_header = True
    current_time = 0

    with path.open(errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if in_header:
                if line.lower().startswith("$timescale"):
                    # VCD headers sometimes put the body on the following
                    # lines, so collect the declaration conservatively.
                    timescale = parse_timescale(line)
                elif line.lower().startswith("$scope"):
                    fields = line.split()
                    if len(fields) >= 3:
                        scope.append(fields[2])
                elif line.lower().startswith("$upscope") and scope:
                    scope.pop()
                elif line.lower().startswith("$var"):
                    fields = line.split()
                    if len(fields) >= 5:
                        width = int(fields[2])
                        identifier = fields[3]
                        name = fields[4]
                        net = Net(tuple(scope), name, width, identifier)
                        nets.append(net)
                        by_id.setdefault(identifier, []).append(net)
                elif line.lower().startswith("$enddefinitions"):
                    in_header = False
                continue

            if line.startswith("#"):
                current_time = int(line[1:])
                end_time = max(end_time, current_time)
                continue
            if line[0] in "01xXzZ" and len(line) > 1:
                for net in by_id.get(line[1:], []):
                    net.change(line[0], current_time)
                continue
            if line[0] in "bBrR":
                fields = line[1:].split()
                if len(fields) == 2:
                    for net in by_id.get(fields[1], []):
                        net.change(fields[0], current_time)

    for net in nets:
        net.add_interval(net.value, end_time - net.last_time)
    top = nets[0].path[0] if nets and nets[0].path else path.stem
    return nets, end_time, timescale, top


def saif_net(net: Net, duration: int) -> str:
    t0, t1 = net.t0, net.t1
    tx = net.tx
    if duration and (t0 + t1 + tx) < duration:
        tx += duration - t0 - t1 - tx
    return (
        f"    (NET {net.name} (T0 {t0}) (T1 {t1}) "
        f"(TX {tx}) (TC {net.toggles}))"
    )


def write_saif(nets: Iterable[Net], duration: int, timescale: str, top: str, path: Path) -> None:
    by_scope: dict[tuple[str, ...], list[Net]] = {}
    for net in nets:
        by_scope.setdefault(net.path, []).append(net)
    lines = [
        "(SAIFILE",
        '  (SAIFVERSION "2.0")',
        '  (DIRECTION "backward")',
        f'  (DATE "{dt.datetime.now(dt.timezone.utc).isoformat()}")',
        '  (VENDOR "OpenAI")',
        '  (PROGRAM_NAME "vcd_to_saif.py")',
        '  (VERSION "1.0")',
        '  (DIVIDER /)',
        f"  (TIMESCALE {timescale})",
        f"  (DURATION {duration})",
        f"  (INSTANCE {top}",
    ]
    scopes = sorted(by_scope, key=lambda item: (len(item), item))
    for scope in scopes:
        if not scope:
            continue
        indent = "    "
        for component in scope[1:]:
            lines.append(f"{indent}(INSTANCE {component}")
            indent += "  "
        for net in by_scope[scope]:
            lines.append(indent + saif_net(net, duration).lstrip())
        lines.extend(")" for _ in scope[1:])
    if tuple() in by_scope:
        for net in by_scope[tuple()]:
            lines.append(saif_net(net, duration))
    lines.append("  )")
    lines.append(")")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vcd", type=Path)
    parser.add_argument("saif", type=Path)
    args = parser.parse_args()
    nets, duration, timescale, top = parse_vcd(args.vcd)
    if not nets:
        raise SystemExit(f"no VCD nets found in {args.vcd}")
    write_saif(nets, duration, timescale, top, args.saif)
    print(f"wrote {args.saif} ({len(nets)} nets, {duration} time units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
