import socket
import pandas as pd


def parse_openbci_band_power_line(line):
    """Parse a single OpenBCI band-power line into channel and band values."""
    line = line.strip()
    if not line:
        return None

    try:
        address_part, data_part = line.split("|", 1)
    except ValueError:
        raise ValueError(f"Unable to split line into address and data: {line}")

    address = address_part.strip()
    if not data_part.strip().lower().startswith("data:"):
        raise ValueError(f"OpenBCI band-power line missing 'Data:' section: {line}")

    channel_str = address.split("/")[-1]
    channel = int(channel_str)

    value_text = data_part.split(":", 1)[1].strip()
    if value_text.startswith("(") and value_text.endswith(")"):
        value_text = value_text[1:-1]

    values = [float(v.strip()) for v in value_text.split(",") if v.strip()]
    return channel, values


def openbci_lines_to_dataframe(lines, num_channels=4, band_names=None):
    """Convert OpenBCI band-power text lines into a pandas DataFrame."""
    if band_names is None:
        band_names = ["delta", "theta", "alpha", "beta", "gamma"]

    if len(band_names) == 0:
        raise ValueError("band_names must contain at least one band label")

    rows = []
    current_window = {}

    for line in lines:
        parsed = parse_openbci_band_power_line(line)
        if parsed is None:
            continue
        channel, values = parsed

        if len(values) != len(band_names):
            raise ValueError(
                f"Expected {len(band_names)} band values, got {len(values)}: {line}"
            )

        current_window[channel] = values
        if len(current_window) == num_channels:
            feature_row = {}
            for ch in sorted(current_window):
                for idx, band in enumerate(band_names):
                    feature_row[f"ch{ch}_{band}"] = current_window[ch][idx]
            rows.append(feature_row)
            current_window = {}

    if current_window:
        print("Warning: dropped incomplete final OpenBCI window")

    return pd.DataFrame(rows)


def load_openbci_band_power_log(path, num_channels=4, band_names=None):
    """Load an OpenBCI band-power log file into a DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return openbci_lines_to_dataframe(lines, num_channels=num_channels, band_names=band_names)


def make_overlapping_windows(df, window_size_seconds=3.0, step_size_seconds=1.5, sample_rate=1.0):
    """Build overlapping averaged windows from sequential OpenBCI rows."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    window_size = max(1, int(round(window_size_seconds * sample_rate)))
    step_size = max(1, int(round(step_size_seconds * sample_rate)))

    if len(df) < window_size:
        return pd.DataFrame(columns=df.columns.tolist() + [
            "window_start_row",
            "window_end_row",
            "window_size_samples",
            "window_start_sec",
            "window_end_sec",
            "window_midpoint_sec",
        ])

    rows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start : start + window_size]
        avg = window.mean(axis=0)
        avg["window_start_row"] = start
        avg["window_end_row"] = start + window_size - 1
        avg["window_size_samples"] = window_size
        avg["window_start_sec"] = start / sample_rate
        avg["window_end_sec"] = (start + window_size - 1) / sample_rate
        avg["window_midpoint_sec"] = (avg["window_start_sec"] + avg["window_end_sec"]) / 2.0
        rows.append(avg)

    return pd.DataFrame(rows)


def load_label_schedule(path):
    """Load a schedule file for automatic labeling."""
    schedule = pd.read_csv(path)
    required = {"start_seconds", "end_seconds", "label"}
    if not required.issubset(set(schedule.columns)):
        raise ValueError("Label schedule file must contain start_seconds,end_seconds,label columns")
    schedule = schedule.copy()
    schedule["start_seconds"] = schedule["start_seconds"].astype(float)
    schedule["end_seconds"] = schedule["end_seconds"].astype(float)
    schedule["label"] = schedule["label"].astype(str)
    return schedule


def assign_labels_to_windows(window_df, schedule_df, default_label=None, label_column="label"):
    """Assign labels to overlapping windows based on a time schedule."""
    if "window_start_row" not in window_df.columns:
        raise ValueError("Window DataFrame must contain window_start_row for automatic labeling")

    if "window_midpoint_sec" not in window_df.columns:
        raise ValueError("Window DataFrame must contain window_midpoint_sec for automatic labeling")

    labels = []
    for time_sec in window_df["window_midpoint_sec"]:
        matches = schedule_df[
            (schedule_df["start_seconds"] <= time_sec)
            & (schedule_df["end_seconds"] > time_sec)
        ]
        if not matches.empty:
            labels.append(matches.iloc[0]["label"])
        else:
            labels.append(default_label)

    result = window_df.copy()
    result[label_column] = labels
    return result


def receive_openbci_band_power_udp(port, num_channels=4, band_names=None, timeout=10.0, max_windows=None):
    """Receive OpenBCI band-power lines over UDP and return raw sample rows."""
    if band_names is None:
        band_names = ["delta", "theta", "alpha", "beta", "gamma"]

    rows = []
    current_window = {}

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("", port))
        if timeout is not None:
            sock.settimeout(timeout)
        print(f"Listening for OpenBCI band-power data on UDP port {port}...")

        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                print("Receive timeout reached, stopping capture.")
                break

            text = data.decode("utf-8", errors="ignore")
            for line in text.splitlines():
                parsed = parse_openbci_band_power_line(line)
                if parsed is None:
                    continue
                channel, values = parsed

                if len(values) != len(band_names):
                    raise ValueError(
                        f"Expected {len(band_names)} band values, got {len(values)}: {line}"
                    )

                current_window[channel] = values
                if len(current_window) == num_channels:
                    feature_row = {}
                    for ch in sorted(current_window):
                        for idx, band in enumerate(band_names):
                            feature_row[f"ch{ch}_{band}"] = current_window[ch][idx]
                    rows.append(feature_row)
                    current_window = {}

                    if max_windows is not None and len(rows) >= max_windows:
                        print(f"Captured {len(rows)} complete windows, stopping.")
                        return pd.DataFrame(rows)

    if current_window:
        print("Warning: dropped incomplete final OpenBCI window")

    return pd.DataFrame(rows)


def stream_openbci_band_power_udp(port, num_channels=4, band_names=None, timeout=10.0, max_windows=None):
    """Receive OpenBCI band-power lines over UDP and yield one feature row at a time."""
    if band_names is None:
        band_names = ["delta", "theta", "alpha", "beta", "gamma"]

    current_window = {}
    windows_emitted = 0

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("", port))
        if timeout is not None:
            sock.settimeout(timeout)
        print(f"Listening for OpenBCI band-power data on UDP port {port}...")

        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                print("Receive timeout reached, stopping capture.")
                break

            text = data.decode("utf-8", errors="ignore")
            for line in text.splitlines():
                parsed = parse_openbci_band_power_line(line)
                if parsed is None:
                    continue
                channel, values = parsed

                if len(values) != len(band_names):
                    raise ValueError(
                        f"Expected {len(band_names)} band values, got {len(values)}: {line}"
                    )

                current_window[channel] = values
                if len(current_window) == num_channels:
                    feature_row = {}
                    for ch in sorted(current_window):
                        for idx, band in enumerate(band_names):
                            feature_row[f"ch{ch}_{band}"] = current_window[ch][idx]
                    yield feature_row
                    windows_emitted += 1
                    current_window = {}

                    if max_windows is not None and windows_emitted >= max_windows:
                        print(f"Captured {windows_emitted} complete windows, stopping.")
                        return

    if current_window:
        print("Warning: dropped incomplete final OpenBCI window")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect EEG band-power data from OpenBCI over UDP")
    parser.add_argument("--recv-port", type=int, required=True, help="UDP port to receive OpenBCI band-power text")
    parser.add_argument("--raw-output-csv", required=True, help="Save raw received OpenBCI sample rows to CSV")
    parser.add_argument("--window-output-csv", help="Save overlapped window feature rows to CSV")
    parser.add_argument("--window-seconds", type=float, default=3.0, help="Window size in seconds for overlapped aggregation")
    parser.add_argument("--step-seconds", type=float, default=1.5, help="Step size in seconds for overlap")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Estimated sample rate in complete channel sets per second")
    parser.add_argument("--num-channels", type=int, default=4, help="Number of channels per OpenBCI window")
    parser.add_argument("--timeout", type=float, default=10.0, help="UDP receive timeout in seconds")
    parser.add_argument("--max-windows", type=int, default=None, help="Maximum number of complete windows to capture")
    parser.add_argument("--label-schedule", help="CSV schedule file with start_seconds,end_seconds,label for automatic labeling")
    parser.add_argument("--default-label", help="Label to assign to windows outside schedule ranges", default=None)
    args = parser.parse_args()

    raw_df = receive_openbci_band_power_udp(
        port=args.recv_port,
        num_channels=args.num_channels,
        timeout=args.timeout,
        max_windows=args.max_windows,
    )
    raw_df.to_csv(args.raw_output_csv, index=False)
    print(f"Saved {len(raw_df)} raw sample rows to {args.raw_output_csv}")

    if args.window_output_csv:
        window_df = make_overlapping_windows(
            raw_df,
            window_size_seconds=args.window_seconds,
            step_size_seconds=args.step_seconds,
            sample_rate=args.sample_rate,
        )

        if args.label_schedule:
            schedule_df = load_label_schedule(args.label_schedule)
            window_df = assign_labels_to_windows(
                window_df,
                schedule_df,
                default_label=args.default_label,
            )

        # Only keep features and label columns for the output CSV
        output_df = window_df.drop(
            columns=[
                c
                for c in [
                    "window_start_row",
                    "window_end_row",
                    "window_size_samples",
                    "window_start_sec",
                    "window_end_sec",
                    "window_midpoint_sec",
                ]
                if c in window_df.columns
            ],
            errors="ignore",
        )
        output_df.to_csv(args.window_output_csv, index=False)
        print(f"Saved {len(output_df)} overlapped window rows to {args.window_output_csv}")
