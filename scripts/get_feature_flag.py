import sys

assert len(sys.argv) > 1, f"{sys.argv[0]} <feature number> [<feature number> ...]"

features = 0
for arg in sys.argv[1:]:
    try:
        bit = int(arg)
    except ValueError:
        print(f"Invalid input '{arg}' for {' '.join(sys.argv)}", file=sys.stderr)
        exit(1)
    features |= (1 << bit)
