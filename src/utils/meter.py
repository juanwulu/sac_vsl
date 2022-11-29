# =============================================================================
# @file   meter.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Monitoring and callback functions"""
from __future__ import annotations

from typing import Dict, ItemsView, Sequence, Union


class AverageMeter:
    """Average metric tracking meter class.
    Attributes:
        name: Name of the metric.
        count: Current total number of updates.
        val: Current value of the metric.
        sum: Current total value of the history metrics.
        avg: Current average value of the history metrics.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def __call__(self) -> float:
        return self.item()

    def __str__(self) -> str:
        fmt_str = '{name} {val:.4f} ({avg:.4f})'
        return fmt_str.format(**self.__dict__)

    def item(self) -> float:
        return self.avg

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """Update meter value.
        Args:
            value: The floating-point value to track.
            n: Number of repeats.
        """
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterGroup(object):
    """Average meter group container.
    Attributes:
        meters: A mapping from string name to average meter class objects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def __getitem__(self, key: str) -> float:
        return self.meters[key].item()

    def __str__(self) -> str:
        return ', '.join(repr(meter) for meter in self.meters.values())

    def items(self) -> ItemsView[str, float]:
        for (key, meter) in self.meters.items():
            yield (key, meter.item())

    def reset(self) -> None:
        self.meters: Dict[str, AverageMeter] = {}

    def update(self, dat: Dict[str, Union[float, Sequence[float]]]) -> None:
        for (name, value) in dat.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter(name)
            if isinstance(value, Sequence):
                for val in value:
                    self.meters[name].update(val)
            else:
                self.meters[name].update(value)
