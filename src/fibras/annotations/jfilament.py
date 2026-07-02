"""Minimal parser for JFilament snake text exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SnakePoint:
    snake_id: int
    point_index: int
    x: float
    y: float
    z_or_slice: float


def parse_jfilament_snakes(path: str | Path) -> list[SnakePoint]:
    """Parse one JFilament `.txt` file containing all `#`-separated snakes."""
    chunks: list[list[str]] = []
    current: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "#":
            chunks.append(current)
            current = []
        else:
            current.append(line)
    chunks.append(current)

    points: list[SnakePoint] = []
    snake_id = 0
    for chunk in chunks:
        snake_points = _parse_snake_chunk(chunk, snake_id)
        if snake_points:
            points.extend(snake_points)
            snake_id += 1
    return points


def _parse_snake_chunk(lines: list[str], snake_id: int) -> list[SnakePoint]:
    points: list[SnakePoint] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            values = [float(field) for field in fields[:5]]
        except ValueError:
            continue
        points.append(
            SnakePoint(
                snake_id=snake_id,
                point_index=int(values[1]),
                x=values[2],
                y=values[3],
                z_or_slice=values[4],
            )
        )
    return points
