"""recording class for loading and preprocessing neuropixels data."""

import numbers
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_TEAL  = "\033[38;2;187;230;228m"
_RESET = "\033[0m"


def _canonical_state(key: str):
    """map a meta.txt state key (case-insensitive) to its canonical name, or None."""
    k = key.lower()
    if k == "awake":
        return "awake"
    if k.startswith("keta"):
        return "ketamine"
    if k.startswith("iso"):
        return "isoflurane"
    if k.startswith("ure"):
        return "urethane"
    return None


def resolve_session_paths(top_dir: Path) -> dict:
    """derive all session file paths from a top-level SpikeGLX session directory."""
    top_dir = Path(top_dir)
    recording_dirs = list(top_dir.glob("*_imec0"))
    nidq_files = list(top_dir.glob("*.nidq.bin"))

    if not recording_dirs:
        raise FileNotFoundError(f"No *_imec0 directory found in {top_dir}")
    if not nidq_files:
        raise FileNotFoundError(f"No *.nidq.bin file found in {top_dir}")

    # discover anesthesia WaveformSequence (any WF csv that isn't the awake one)
    wf_anesthesia = [
        f for f in top_dir.glob("WaveformSequence_*.csv")
        if "awake" not in f.name.lower()
    ]

    return {
        "recording_dir": recording_dirs[0],
        "nidq_file": nidq_files[0],
        "waveform_csv_awake": top_dir / "WaveformSequence_awake.csv",
        "waveform_csv_anesthesia": wf_anesthesia[0] if wf_anesthesia else None,
    }


S2V_AP  = 2.34375e-06
S2V_LFP = 4.6875e-06

NC = 384

CHANNEL_GRID = {
    1: dict(DX=16, X0=11, DY=20, Y0=20),
    2: dict(DX=32, X0=27, DY=15, Y0=20),
    "NPultra": dict(DX=6, X0=0, DY=6, Y0=0),
}

MAJOR_VERSION = {
    "3A": 1,
    "3B2": 1,
    "3B1": 1,
    "NP2.1": 2,
    "NP2.4": 2.4,
    "NPultra": "NPultra",
}

DEFAULT_CONFIG = {
    "spike_processing": {
        "refractory_period_ms": 1.0,
        "quality_filter": "good",
    },
    "files": {
        "ks_dir": "ks",
        "user_meta": "meta.txt",
        "area_depths": "area_depths.csv",
    },
}


def _xy2rc(x, y, version=1):
    """convert um coordinates to row/col indices."""
    version = np.floor(version) if isinstance(version, numbers.Number) else version
    grid = CHANNEL_GRID[version]
    col = (x - grid["X0"]) / grid["DX"]
    row = (y - grid["Y0"]) / grid["DY"]
    return {"col": col, "row": row}


def _rc2xy(row, col, version=1):
    """convert row/col indices to um coordinates."""
    version = np.floor(version) if isinstance(version, numbers.Number) else version
    grid = CHANNEL_GRID[version]
    x = col * grid["DX"] + grid["X0"]
    y = row * grid["DY"] + grid["Y0"]
    return {"x": x, "y": y}


def _dense_layout(version=1, nshank=1):
    """return dense channel layout for neuropixel probes."""
    ch = {
        "ind": np.arange(NC),
        "row": np.floor(np.arange(NC) / 2),
        "shank": np.zeros(NC),
    }

    if version == 1:
        ch.update({"col": np.tile(np.array([2, 0, 3, 1]), int(NC / 4))})
    elif version == "NPultra":
        ch.update({"row": np.floor(np.arange(NC) / 8)})
        ch.update({"col": np.tile(np.arange(8), int(NC / 8))})
    elif np.floor(version) == 2 and nshank == 1:
        ch.update({"col": np.tile(np.array([0, 1]), int(NC / 2))})
    elif np.floor(version) == 2 and nshank == 4:
        shank_row = np.tile(np.arange(NC / 16), (2, 1)).T[:, np.newaxis].flatten()
        shank_row = np.tile(shank_row, 8)
        shank_row += (
            np.tile(
                np.array([0, 0, 1, 1, 0, 0, 1, 1])[:, np.newaxis], (1, int(NC / 8))
            ).flatten()
            * 24
        )
        ch.update(
            {
                "col": np.tile(np.array([0, 1]), int(NC / 2)),
                "shank": np.tile(
                    np.array([0, 1, 0, 1, 2, 3, 2, 3])[:, np.newaxis], (1, int(NC / 8))
                ).flatten(),
                "row": shank_row,
            }
        )
    ch.update(_rc2xy(ch["row"], ch["col"], version=version))
    return ch


def _adc_shifts(version=1, nc=NC):
    """calculate ADC timing shifts for each channel."""
    if version == 1 or version == "NPultra":
        adc_channels = 12
        n_cycles = 13
    elif np.floor(version) == 2:
        adc_channels = n_cycles = 16
    adc = np.floor(np.arange(NC) / (adc_channels * 2)) * 2 + np.mod(np.arange(NC), 2)
    sample_shift = np.zeros_like(adc)
    for a in adc:
        sample_shift[adc == a] = np.arange(adc_channels) / n_cycles
    return sample_shift[:nc], adc[:nc]


def _trace_header(version=1, nshank=1):
    """return default channel map for dense layouts."""
    h = _dense_layout(version=version, nshank=nshank)
    h["sample_shift"], h["adc"] = _adc_shifts(version=version)
    return h


def _get_neuropixel_version_from_meta(md: dict) -> str:
    """get neuropixel version tag from metadata dictionary."""
    if "typeEnabled" in md.keys():
        return "3A"
    prb_type = md.get("imDatPrb_type")
    if prb_type == 0:
        if "imDatPrb_port" in md.keys() and "imDatPrb_slot" in md.keys():
            return "3B2"
        else:
            return "3B1"
    elif prb_type == 21 or prb_type == 1030:
        return "NP2.1"
    elif prb_type == 24 or prb_type == 2013:
        return "NP2.4"
    elif prb_type == 1100:
        return "NPultra"
    return None


def _get_neuropixel_major_version_from_meta(md: dict):
    """get major version number (1, 2, 2.4, or 'NPultra') from metadata."""
    version = _get_neuropixel_version_from_meta(md)
    if version is not None:
        return MAJOR_VERSION.get(version)
    return None


def _map_channels_from_meta(meta_data: dict) -> dict:
    """extract channel positions from metadata string."""
    if "snsShankMap" in meta_data.keys():
        chmap = re.findall(r"([0-9]*:[0-9]*:[0-9]*:[0-9]*)", meta_data["snsShankMap"])
        key_names = {"shank": 0, "col": 1, "row": 2, "flag": 3}
    elif "snsGeomMap" in meta_data.keys():
        chmap = re.findall(r"([0-9]*:[0-9]*:[0-9]*:[0-9]*)", meta_data["snsGeomMap"])
        key_names = {"shank": 0, "x": 1, "y": 2, "flag": 3}
    else:
        return None
    if not chmap:
        return {"shank": None, "col": None, "row": None, "flag": None}
    chmap = np.array([np.float32(cm.split(":")) for cm in chmap])
    return {k: chmap[:, v] for (k, v) in key_names.items()}


def _split_geometry_into_shanks(th: dict, meta_data: dict) -> dict:
    """reduce geometry to specific shank for NP2.4 probes."""
    if "NP2.4_shank" in meta_data.keys():
        shank_idx = np.where(th["shank"] == int(meta_data["NP2.4_shank"]))[0]
        th = {key: th[key][shank_idx] for key in th.keys()}
    return th


def _geometry_from_meta(meta_data: dict, nc: int = 384) -> dict:
    """get probe geometry from metadata, with fallback to default layout."""
    cm = _map_channels_from_meta(meta_data)
    major_version = _get_neuropixel_major_version_from_meta(meta_data)

    if cm is None or all(map(lambda x: x is None, cm.values())):
        # fallback to default layout
        if major_version is None:
            return None
        th = _trace_header(version=major_version)
        th["flag"] = th["x"] * 0 + 1.0
        return th

    th = cm.copy()
    if "x" in cm.keys():
        if major_version == 1:
            th["x"] = 70 - th["x"]
        th["y"] += 20
        th.update(_xy2rc(th["x"], th["y"], version=major_version))
    else:
        if major_version == 1:
            th["col"] = -cm["col"] * 2 + 2 + np.mod(cm["row"], 2)
        th.update(_rc2xy(th["row"], th["col"], version=major_version))

    th["sample_shift"], th["adc"] = _adc_shifts(
        version=major_version, nc=th["col"].size
    )
    th = _split_geometry_into_shanks(th, meta_data)
    th["ind"] = np.arange(th["col"].size)

    # sort by shank, row, col
    sort_keys = np.c_[-th["col"], th["row"], th["shank"]]
    inds = np.lexsort(sort_keys.T)
    th = {k: v[inds] for k, v in th.items()}
    return th


class Recording:
    """loads spike sorting results and probe geometry for a single neuropixels recording."""

    def __init__(self, recording_dir: Path, config: dict):
        self.recording_dir = Path(recording_dir)
        self.config = self._validate_config(config)

        self.probe_num: int = self._extract_probe_num()

        self.paths: dict = {}
        self.LFmetaDict: dict = {}
        self.APmetaDict: dict = {}
        self.probeVersion: str = None
        self.probeGeometry: dict = {}
        self.surfaceChan: int = None
        self.region: str = None
        self.stateTimes: dict = {}  # {state: (start_min, end_min)}, end=np.inf if "end"
        self.clusterInfo: pd.DataFrame = None
        self.unitSpikes: Dict[int, np.ndarray] = {}
        self.areaDepths: dict = {}

        self._get_paths()
        self._load_meta()
        self._load_probe_geometry()
        self._load_ks_data()
        self._load_area_depths()
        self._filter_refractory_violations()
        self._assign_layers()

    def __repr__(self) -> str:
        total_spikes = sum(len(s) for s in self.unitSpikes.values())

        layer_counts = self.clusterInfo["layer"].value_counts().to_dict()
        layer_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(layer_counts.items()) if k is not None
        )
        unassigned = self.clusterInfo["layer"].isna().sum()

        n_channels = len(self.probeGeometry.get("x", []))

        lines = [
            f"Recording(probe={self.probe_num})",
            f"  Directory:     {self.recording_dir.name}",
            f"  Probe version: {self.probeVersion}",
            f"  Channels:      {n_channels}",
            f"  Surface chan:  {self.surfaceChan}",
            f"  Region:        {self.region}",
            f"  Clusters:      {len(self.clusterInfo)}",
            f"  Units:         {len(self.unitSpikes)}",
            f"  Total spikes:  {total_spikes:,}",
            f"  Layer depths:  {len(self.areaDepths)} layers defined",
            f"  Layer assign:  {layer_str}",
            f"  State times:   "
            + (
                ", ".join(
                    f"{k} {v[0]:.0f}–{'end' if np.isinf(v[1]) else f'{v[1]:.0f}'} min"
                    for k, v in self.stateTimes.items()
                )
                if self.stateTimes
                else "not specified"
            ),
        ]
        if unassigned > 0:
            lines.append(f"  Unassigned:    {unassigned}")

        return "\n".join(lines)

    def _validate_config(self, config: dict) -> dict:
        validated = {}
        validated["spike_processing"] = {
            **DEFAULT_CONFIG["spike_processing"],
            **config.get("spike_processing", {}),
        }
        validated["files"] = {
            **DEFAULT_CONFIG["files"],
            **config.get("files", {}),
        }
        return validated

    def _extract_probe_num(self) -> int:
        folder_name = self.recording_dir.name
        match = re.search(r"imec(\d+)$", folder_name)
        if not match:
            raise ValueError(
                f"Cannot extract probe number from '{folder_name}'. "
                f"Expected folder name ending in 'imec0', 'imec1', etc."
            )
        return int(match.group(1))

    def _get_paths(self) -> None:
        files_config = self.config["files"]

        self.paths = {
            "lfMetaPath": None,
            "apMetaPath": None,
            "userMetaPath": self.recording_dir / files_config["user_meta"],
            "rawApBinPath": None,
            "rawLfBinPath": None,
            "ksPath": self.recording_dir / files_config["ks_dir"],
            "areaDepthsPath": self.recording_dir / files_config["area_depths"],
        }

        for f in self.recording_dir.iterdir():
            if f.is_file():
                if f.name.endswith(".lf.meta"):
                    self.paths["lfMetaPath"] = f
                elif f.name.endswith(".ap.meta"):
                    self.paths["apMetaPath"] = f
                elif f.name.endswith(".ap.bin") or f.name.endswith(".ap.cbin"):
                    self.paths["rawApBinPath"] = f
                elif f.name.endswith(".lf.bin") or f.name.endswith(".lf.cbin"):
                    self.paths["rawLfBinPath"] = f

        if self.paths["apMetaPath"] is None:
            raise FileNotFoundError(
                f"AP metadata file (.ap.meta) not found in {self.recording_dir}"
            )
        if not self.paths["ksPath"].exists():
            raise FileNotFoundError(
                f"Kilosort directory not found: {self.paths['ksPath']}"
            )
        if not self.paths["userMetaPath"].exists():
            raise FileNotFoundError(
                f"User metadata file not found: {self.paths['userMetaPath']}\n"
                f"Create this file with format: 'sur <channel_number>'"
            )
        if not self.paths["areaDepthsPath"].exists():
            raise FileNotFoundError(
                f"Layer boundaries file not found: {self.paths['areaDepthsPath']}\n"
                f"Create this file with columns: Layer, Start, End"
            )

    def _load_meta(self) -> None:
        self.APmetaDict = self._parse_meta_file(self.paths["apMetaPath"])

        if self.paths["lfMetaPath"] is not None:
            self.LFmetaDict = self._parse_meta_file(self.paths["lfMetaPath"])

        self.probeVersion = _get_neuropixel_version_from_meta(self.APmetaDict)
        self._load_user_meta()

    def _parse_meta_file(self, path: Path) -> dict:
        """parse SpikeGLX meta file into dictionary."""
        meta_dict = {}
        with path.open() as f:
            for line in f.read().splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.startswith("~"):
                    key = key[1:]
                key = key.strip()
                value = value.strip()

                if value and re.fullmatch(r"[0-9,.]*", value) and value.count(".") < 2:
                    parsed = [float(val) for val in value.split(",")]
                    value = parsed[0] if len(parsed) == 1 else parsed

                meta_dict[key] = value
        return meta_dict

    def _load_user_meta(self) -> None:
        with self.paths["userMetaPath"].open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                key, value = parts[0], parts[1]
                if key == "sur":
                    self.surfaceChan = int(value)
                elif key == "region":
                    self.region = value
                else:
                    canonical = _canonical_state(key)
                    if canonical is not None:
                        # expect: <key> <start> - <end>  (times in minutes; "end" → np.inf)
                        time_parts = [p for p in parts[1:] if p != "-"]
                        start = float(time_parts[0])
                        end = np.inf if len(time_parts) < 2 or time_parts[1] == "end" else float(time_parts[1])
                        self.stateTimes[canonical] = (start, end)

        if self.surfaceChan is None:
            raise ValueError(
                f"Required field 'sur' not found in {self.paths['userMetaPath']}\n"
                f"Add a line like: sur 325"
            )

    def _load_probe_geometry(self) -> None:
        self.probeGeometry = _geometry_from_meta(self.APmetaDict)
        if self.probeGeometry is None:
            raise ValueError(
                f"Could not determine probe geometry from metadata.\n"
                f"Probe version: {self.probeVersion}"
            )

    def _load_ks_data(self) -> None:
        ks_dir = self.paths["ksPath"]

        cluster_info_file = ks_dir / "cluster_info.tsv"
        if not cluster_info_file.exists():
            raise FileNotFoundError(f"cluster_info.tsv not found in {ks_dir}")
        self.clusterInfo = pd.read_csv(cluster_info_file, sep="\t")

        # filter by unit quality early — all downstream data only covers kept units
        quality = self.config["spike_processing"]["quality_filter"]
        if quality == "good":
            keep = self.clusterInfo["group"] == "good"
        elif quality == "mua":
            keep = self.clusterInfo["group"].isin(("good", "mua"))
        else:  # "all"
            keep = pd.Series(True, index=self.clusterInfo.index)
        self.clusterInfo = self.clusterInfo[keep].reset_index(drop=True)

        good_ids = set(self.clusterInfo["cluster_id"])

        spike_times_file = ks_dir / "spike_times.npy"
        if not spike_times_file.exists():
            raise FileNotFoundError(f"spike_times.npy not found in {ks_dir}")
        spike_times = np.load(spike_times_file).flatten()

        # convert to seconds if needed
        spike_times_sec_file = ks_dir / "spike_times_seconds.npy"
        if spike_times_sec_file.exists():
            spike_times = np.load(spike_times_sec_file).flatten()
        else:
            sample_rate = float(self.APmetaDict.get("imSampRate", 30000.0))
            spike_times = spike_times / sample_rate
            np.save(spike_times_sec_file, spike_times)

        spike_clusters_file = ks_dir / "spike_clusters.npy"
        if not spike_clusters_file.exists():
            raise FileNotFoundError(f"spike_clusters.npy not found in {ks_dir}")
        spike_clusters = np.load(spike_clusters_file).flatten()

        keep_mask = np.isin(spike_clusters, list(good_ids))
        spike_clusters = spike_clusters[keep_mask]
        spike_times = spike_times[keep_mask]

        # organize spikes by unit using np.split (faster than dict loop)
        sort_idx = np.argsort(spike_clusters)
        sorted_clusters = spike_clusters[sort_idx]
        sorted_times = spike_times[sort_idx]
        unique_clusters, split_indices = np.unique(sorted_clusters, return_index=True)
        spike_arrays = np.split(sorted_times, split_indices[1:])
        self.unitSpikes = dict(zip(unique_clusters, spike_arrays))


    def _load_area_depths(self) -> None:
        path = self.paths["areaDepthsPath"]
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        layer_col = next((c for c in df.columns if "layer" in c.lower()), None)
        start_col = next((c for c in df.columns if "start" in c.lower()), None)
        end_col = next((c for c in df.columns if "end" in c.lower()), None)

        if not all([layer_col, start_col, end_col]):
            raise ValueError(
                f"Missing required columns in {path}\n"
                f"Found: {list(df.columns)}\n"
                f"Required: Layer, Start, End"
            )

        self.areaDepths = {}
        for _, row in df.iterrows():
            layer_name = str(row[layer_col]).strip()
            if not layer_name or layer_name == "nan":
                continue

            # normalize layer name: L1 → "1", L2/3 → "2/3"
            if layer_name.startswith("L") and len(layer_name) > 1:
                layer_name = layer_name[1:]

            self.areaDepths[layer_name] = (float(row[start_col]), float(row[end_col]))

    def _filter_refractory_violations(self) -> None:
        refractory_ms = self.config["spike_processing"]["refractory_period_ms"]
        refractory_sec = refractory_ms * 1e-3

        total_removed = 0
        for unit_id, spike_times in self.unitSpikes.items():
            if len(spike_times) <= 1:
                continue
            isis = np.diff(spike_times)
            keep_mask = np.ones(len(spike_times), dtype=bool)
            keep_mask[1:] = isis > refractory_sec
            n_removed = len(spike_times) - np.sum(keep_mask)
            total_removed += n_removed
            self.unitSpikes[unit_id] = spike_times[keep_mask]


    def _assign_layers(self) -> None:
        surface_um = self.surfaceChan * 10  # 10µm spacing
        self.clusterInfo["brain_depth"] = surface_um - self.clusterInfo["depth"]

        self.clusterInfo["layer"] = None

        sorted_layers = sorted(self.areaDepths.items(), key=lambda x: x[1][0])

        for layer_name, (start_depth, end_depth) in sorted_layers:
            unassigned = self.clusterInfo["layer"].isna()
            mask = unassigned & (self.clusterInfo["brain_depth"] >= start_depth) & (
                self.clusterInfo["brain_depth"] < end_depth
            )
            self.clusterInfo.loc[mask, "layer"] = layer_name


        self.clusterInfo.to_csv(
            self.paths["ksPath"] / "cluster_info.tsv", sep="\t", index=False
        )

    def _get_raw_reader(self, band: str = "ap") -> np.memmap:
        """memory-map raw binary data file. band: 'ap' or 'lf'."""
        if band == "ap":
            path = self.paths["rawApBinPath"]
            meta = self.APmetaDict
        else:
            path = self.paths["rawLfBinPath"]
            meta = self.LFmetaDict

        if path is None or not path.exists():
            raise FileNotFoundError(f"Raw {band} binary file not found")

        if path.suffix == ".cbin":
            raise NotImplementedError(
                f"Compressed .cbin files not supported. Decompress first."
            )

        num_channels = int(meta["nSavedChans"])
        num_samples = int(int(meta["fileSizeBytes"]) / (2 * num_channels))

        data = np.memmap(
            path,
            dtype="int16",
            mode="r",
            shape=(num_channels, num_samples),
            order="F",
        )
        return data

    def _samples_to_volts(self, data: np.ndarray, band: str = "ap") -> np.ndarray:
        """convert raw int16 samples to volts."""
        s2v = S2V_AP if band == "ap" else S2V_LFP
        return data.astype(np.float32) * s2v
