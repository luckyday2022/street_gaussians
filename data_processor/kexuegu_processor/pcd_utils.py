import io
from typing import NamedTuple, Tuple

import numpy as np


class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    mask: np.ndarray


_PLY_TYPE_TO_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _parse_header(path: str) -> Tuple[str, int, list, int]:
    with open(path, "rb") as f:
        first = f.readline().decode("ascii").strip()
        if first != "ply":
            raise RuntimeError(f"Invalid PLY file (missing 'ply'): {path}")

        fmt_line = f.readline().decode("ascii").strip().split()
        if len(fmt_line) < 3 or fmt_line[0] != "format":
            raise RuntimeError(f"Invalid PLY format line: {path}")
        fmt = fmt_line[1]

        vertex_count = 0
        in_vertex = False
        properties = []

        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in PLY header: {path}")
            line_str = line.decode("ascii").strip()

            if line_str.startswith("element "):
                toks = line_str.split()
                in_vertex = toks[1] == "vertex"
                if in_vertex:
                    vertex_count = int(toks[2])

            elif line_str.startswith("property ") and in_vertex:
                toks = line_str.split()
                if len(toks) != 3 or toks[1] == "list":
                    raise RuntimeError(f"Unsupported PLY property in {path}: {line_str}")
                ply_type, name = toks[1], toks[2]
                if ply_type not in _PLY_TYPE_TO_DTYPE:
                    raise RuntimeError(f"Unsupported PLY type '{ply_type}' in {path}")
                properties.append((name, ply_type))

            elif line_str == "end_header":
                data_offset = f.tell()
                break

    if vertex_count <= 0:
        raise RuntimeError(f"No vertex element found in {path}")
    if len(properties) == 0:
        raise RuntimeError(f"No vertex properties found in {path}")

    return fmt, vertex_count, properties, data_offset


def _dtype_from_properties(properties: list, little_endian: bool = True) -> np.dtype:
    endian = "<" if little_endian else ">"
    fields = []
    for name, ply_type in properties:
        fields.append((name, endian + _PLY_TYPE_TO_DTYPE[ply_type]))
    return np.dtype(fields)


def _read_vertex_structured(path: str) -> np.ndarray:
    fmt, vertex_count, properties, data_offset = _parse_header(path)

    if fmt == "binary_little_endian":
        dtype = _dtype_from_properties(properties, little_endian=True)
        with open(path, "rb") as f:
            f.seek(data_offset)
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
        return data

    if fmt == "ascii":
        dtype = _dtype_from_properties(properties, little_endian=True)
        with open(path, "rb") as f:
            f.seek(data_offset)
            text = f.read().decode("ascii")
        cols = np.loadtxt(io.StringIO(text))
        if cols.ndim == 1:
            cols = cols[None]

        out = np.zeros((cols.shape[0],), dtype=dtype)
        for i, (name, _) in enumerate(properties):
            out[name] = cols[:, i]
        return out

    raise RuntimeError(f"Unsupported PLY format: {fmt} ({path})")


def fetch_ply(path: str) -> BasicPointCloud:
    vertices = _read_vertex_structured(path)

    if not all(k in vertices.dtype.names for k in ("x", "y", "z")):
        raise RuntimeError(f"PLY missing xyz fields: {path}")

    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)

    if all(k in vertices.dtype.names for k in ("red", "green", "blue")):
        colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        colors = np.zeros((points.shape[0], 3), dtype=np.float32)

    if "mask" in vertices.dtype.names:
        mask = vertices["mask"].astype(np.bool_)
    else:
        mask = np.ones((points.shape[0],), dtype=np.bool_)

    return BasicPointCloud(points=points, colors=colors, mask=mask)


def store_ply(path: str, xyz: np.ndarray, rgb: np.ndarray, mask: np.ndarray) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(f"xyz should be [N, 3], got {xyz.shape}")

    if rgb.max() <= 1.0 and rgb.min() >= 0.0:
        rgb = np.clip(rgb * 255.0, 0.0, 255.0)

    rgb = rgb.astype(np.uint8)
    mask = mask.reshape(-1).astype(np.uint8)

    vertices = np.empty(
        xyz.shape[0],
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("mask", "u1"),
            ]
        ),
    )
    vertices["x"] = xyz[:, 0].astype(np.float32)
    vertices["y"] = xyz[:, 1].astype(np.float32)
    vertices["z"] = xyz[:, 2].astype(np.float32)
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]
    vertices["mask"] = mask

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar mask\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def read_xyz_from_ply(path: str) -> np.ndarray:
    vertices = _read_vertex_structured(path)
    if not all(k in vertices.dtype.names for k in ("x", "y", "z")):
        raise RuntimeError(f"PLY missing xyz fields: {path}")
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    return xyz.astype(np.float32)
