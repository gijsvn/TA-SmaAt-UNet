import torch
import math

_MONTHS = {
    b'JAN': 1, b'FEB': 2, b'MAR': 3, b'APR': 4,
    b'MAY': 5, b'JUN': 6, b'JUL': 7, b'AUG': 8,
    b'SEP': 9, b'OCT': 10, b'NOV': 11, b'DEC': 12,
}


_DAYS_BEFORE_MONTH = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def _is_leap(year: int) -> bool:
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


def _day_of_year(day: int, month: int, year: int) -> int:
    doy = _DAYS_BEFORE_MONTH[month - 1] + day
    if month > 2 and _is_leap(year):
        doy += 1
    return doy

def timestamp_to_vector(timestamp: str) -> torch.Tensor:
    day = (timestamp[0] - 48) * 10 + (timestamp[1] - 48)

    # month: positions 3-5 (3-letter abbrev)
    month = _MONTHS[timestamp[3:6]]

    # year: positions 7-10
    year = ((timestamp[7] - 48) * 1000 +
            (timestamp[8] - 48) * 100 +
            (timestamp[9] - 48) * 10 +
            (timestamp[10] - 48))

    # time: HH:MM:SS.mmm
    # HH: 12-13
    hour = (timestamp[12] - 48) * 10 + (timestamp[13] - 48)
    # MM: 15-16
    minute = (timestamp[15] - 48) * 10 + (timestamp[16] - 48)
    # SS: 18-19
    second = (timestamp[18] - 48) * 10 + (timestamp[19] - 48)
    # mmm: 21-23
    ms = ((timestamp[21] - 48) * 100 +
          (timestamp[22] - 48) * 10 +
          (timestamp[23] - 48))

    # --- time-of-day as fraction in [0, 1) ---
    total_ms = (((hour * 60 + minute) * 60 + second) * 1000 + ms)
    tod = total_ms / (24.0 * 60.0 * 60.0 * 1000.0)  # fraction of day

    # --- day-of-year ---
    doy = _day_of_year(day, month, year)
    n_y = 366 if _is_leap(year) else 365

    # --- angles ---
    tod_angle = 2.0 * math.pi * tod
    toy_angle = 2.0 * math.pi * (doy / n_y)

    tod_sin = math.sin(tod_angle)
    tod_cos = math.cos(tod_angle)
    toy_sin = math.sin(toy_angle)
    toy_cos = math.cos(toy_angle)

    return torch.tensor(
        [tod_sin, tod_cos, toy_sin, toy_cos]
    )
