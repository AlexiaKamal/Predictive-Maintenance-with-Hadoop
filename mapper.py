#!/usr/bin/env python3
import sys

for line in sys.stdin:
    values = line.strip().split()

    if len(values) < 26:
        continue

    engine_id = values[0]
    cycle = int(values[1])
    sensors = values[2:10]

    print(f"{engine_id}\t{cycle},{','.join(sensors)}")