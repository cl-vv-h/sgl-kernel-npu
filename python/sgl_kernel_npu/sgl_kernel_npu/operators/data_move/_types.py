"""Transfer enums shared by public operators and the historical kvcacheio path."""

from enum import Enum


class TransferDirection(Enum):
    H2D = 1
    D2H = 2


class TransferFlag(Enum):
    FAST2D = 2


def direction_value(direction):
    value = direction.value if isinstance(direction, TransferDirection) else direction
    if value not in (1, 2):
        raise ValueError("direction must be H2D (1) or D2H (2)")
    return value


def flags_value(flags):
    value = flags.value if isinstance(flags, TransferFlag) else flags
    if not isinstance(value, int) or (value & 2) != 2:
        raise ValueError("transfer requires FAST2D (flags & 2 == 2)")
    return value
