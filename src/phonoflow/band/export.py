"""Export structured phonon band data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from phonoflow.band.data import BandData, band_data_to_metadata
from phonoflow.band.labels import format_band_label, normalize_band_label


def export_phonon_band_data(band_data: BandData, outdir: Path) -> dict[str, str]:
    """Export band data in wide, long, text, segment, and metadata formats."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "phonon_band.csv"
    long_csv_path = outdir / "phonon_band_long.csv"
    dat_path = outdir / "phonon_band.dat"
    segments_path = outdir / "phonon_band_segments.json"
    metadata_path = outdir / "phonon_band_metadata.json"

    _write_band_csv(band_data, csv_path)
    _write_band_long_csv(band_data, long_csv_path)
    _write_band_dat(band_data, dat_path)
    _write_json(_segments_payload(band_data), segments_path)
    _write_json(band_data_to_metadata(band_data), metadata_path)
    return {
        "band_data": dat_path.name,
        "band_csv": csv_path.name,
        "band_long_csv": long_csv_path.name,
        "band_segments": segments_path.name,
        "band_metadata": metadata_path.name,
    }


def _write_band_csv(band_data: BandData, path: Path) -> None:
    header = [
        "segment_index",
        "q_index_global",
        "q_index_in_segment",
        "distance",
        "qx",
        "qy",
        "qz",
    ]
    header.extend([f"branch_{index + 1}_THz" for index in range(band_data.n_branches)])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        global_index = 0
        for segment in band_data.segments:
            for local_index in range(segment.nqpoint):
                row = [
                    segment.index,
                    global_index,
                    local_index,
                    float(segment.distances[local_index]),
                    *[float(value) for value in segment.qpoints[local_index]],
                    *[float(value) for value in segment.frequencies[local_index]],
                ]
                writer.writerow(row)
                global_index += 1


def _write_band_long_csv(band_data: BandData, path: Path) -> None:
    header = [
        "segment_index",
        "q_index_global",
        "q_index_in_segment",
        "distance",
        "qx",
        "qy",
        "qz",
        "branch_index",
        "frequency_THz",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        global_index = 0
        for segment in band_data.segments:
            for local_index in range(segment.nqpoint):
                qpoint = [float(value) for value in segment.qpoints[local_index]]
                for branch_index, frequency in enumerate(segment.frequencies[local_index], start=1):
                    writer.writerow(
                        [
                            segment.index,
                            global_index,
                            local_index,
                            float(segment.distances[local_index]),
                            *qpoint,
                            branch_index,
                            float(frequency),
                        ]
                    )
                global_index += 1


def _write_band_dat(band_data: BandData, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# PhonoFlow phonon band data\n")
        handle.write(f"# frequency unit: {band_data.frequency_unit}\n")
        branch_columns = " ".join(f"branch_{index + 1}" for index in range(band_data.n_branches))
        handle.write(f"# columns: distance {branch_columns}\n")
        for segment in band_data.segments:
            handle.write(
                f"# segment {segment.index}: {_plain_label(segment.start_label)} -> "
                f"{_plain_label(segment.end_label)}\n"
            )
            for distance, frequencies in zip(segment.distances, segment.frequencies, strict=False):
                values = " ".join(f"{float(value):.10f}" for value in frequencies)
                handle.write(f"{float(distance):.10f} {values}\n")
            handle.write("\n")


def _segments_payload(band_data: BandData) -> dict[str, Any]:
    return {
        "segments": [
            {
                "segment_index": segment.index,
                "start_label": _plain_label(segment.start_label),
                "end_label": _plain_label(segment.end_label),
                "start_tick_label": format_band_label(segment.start_label),
                "end_tick_label": format_band_label(segment.end_label),
                "distance_start": float(segment.distances[0]),
                "distance_end": float(segment.distances[-1]),
                "nqpoint": segment.nqpoint,
            }
            for segment in band_data.segments
        ]
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _plain_label(label: str) -> str:
    normalized = normalize_band_label(label)
    if normalized == "\N{GREEK CAPITAL LETTER GAMMA}":
        return "Gamma"
    return str(label).strip()
