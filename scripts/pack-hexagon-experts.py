#!/usr/bin/env python3
"""Pack routed experts from a single GGUF for the Hexagon MoE cache."""

import argparse
import ctypes as C
import json
import os
import re
import stat
import sys
import tempfile
import zlib
from pathlib import Path
from typing import NamedTuple

LAYOUT = "hexagon-tiled-v1"
COMPACT_LAYOUT = "hexagon-q3k-compact-v1"
INDEX = "index.json"
MANIFEST = "manifest.json"
TENSOR_NAME = re.compile(r"blk\.(\d+)\.ffn_(?:gate_up|gate|up|down)_exps\.weight")
CHUNK_EXPERTS = 16
DIRECT_ALIGNMENT = 4096
MAX_INDEX_BYTES = 8 * 1024 * 1024


class GGUFParams(C.Structure):
    _fields_ = [("no_alloc", C.c_bool), ("ctx", C.POINTER(C.c_void_p))]


class ExpertTensor(NamedTuple):
    offset: int
    kind: int
    ne0: int
    ne1: int
    count: int
    raw_size: int
    packed_size: int


def bind(lib, name, restype, *argtypes):
    function = getattr(lib, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


def open_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"expected a regular file: {path}")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def file_stamp(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def read_model(base, libc, source, model_size, pack):
    void = C.c_void_p
    i64 = C.c_int64
    fdopen = bind(libc, "fdopen", void, C.c_int, C.c_char_p)
    fclose = bind(libc, "fclose", C.c_int, void)
    init = bind(base, "gguf_init_from_file_ptr", void, void, GGUFParams)
    free = bind(base, "gguf_free", None, void)
    get_count = bind(base, "gguf_get_n_tensors", i64, void)
    get_name = bind(base, "gguf_get_tensor_name", C.c_char_p, void, i64)
    get_shape = bind(base, "gguf_get_tensor_ne", C.POINTER(i64), void, i64)
    get_type = bind(base, "gguf_get_tensor_type", C.c_int, void, i64)
    get_size = bind(base, "gguf_get_tensor_size", C.c_size_t, void, i64)
    get_offset = bind(base, "gguf_get_tensor_offset", C.c_size_t, void, i64)
    data_offset = bind(base, "gguf_get_data_offset", C.c_size_t, void)
    find_key = bind(base, "gguf_find_key", i64, void, C.c_char_p)
    key_type = bind(base, "gguf_get_kv_type", C.c_int, void, i64)
    get_u16 = bind(base, "gguf_get_val_u16", C.c_uint16, void, i64)

    fd = os.dup(source.fileno())
    model = fdopen(fd, b"rb")
    if not model:
        error = C.get_errno()
        os.close(fd)
        raise OSError(error, os.strerror(error))
    try:
        ctx = init(model, GGUFParams(True, None))
    finally:
        fclose(model)
    if not ctx:
        raise ValueError("cannot read source GGUF metadata")

    layers = {}
    unsupported = {}
    try:
        split = find_key(ctx, b"split.count")
        if split >= 0:
            if key_type(ctx, split) != 2:  # GGUF_TYPE_UINT16
                raise ValueError("invalid GGUF split.count metadata")
            if get_u16(ctx, split) > 1:
                raise ValueError("merge split GGUF files before packing")

        offset = data_offset(ctx)
        for i in range(get_count(ctx)):
            name = get_name(ctx, i).decode()
            match = TENSOR_NAME.fullmatch(name)
            if not match:
                continue
            layer = int(match[1])
            ne0, ne1, count, ne3 = get_shape(ctx, i)[:4]
            kind = get_type(ctx, i)
            start, size = offset + get_offset(ctx, i), get_size(ctx, i)
            if size <= 0 or start + size > model_size:
                raise ValueError(f"invalid GGUF tensor extent: {name}")
            packed_size = pack(None, None, kind, ne0, ne1)
            if not packed_size or not 0 < count <= 0x7fffffff or ne3 != 1 or size % count:
                unsupported[layer] = name
                continue
            tensors = layers.setdefault(layer, {})
            if name in tensors:
                raise ValueError(f"duplicate tensor: {name}")
            tensors[name] = ExpertTensor(start, kind, ne0, ne1, count, size // count, packed_size)
    finally:
        free(ctx)

    for layer, name in unsupported.items():
        layers.pop(layer, None)
        print(f"skipping layer {layer}: unsupported tensor {name}", file=sys.stderr)  # noqa: NP100
    if not layers:
        raise ValueError("GGUF has no packable routed expert layers")
    for layer, tensors in layers.items():
        if len({tensor.count for tensor in tensors.values()}) != 1:
            raise ValueError(f"inconsistent expert count in layer {layer}")
    return layers


def read_json(path):
    with open_regular(path) as file:
        data = file.read(MAX_INDEX_BYTES + 1)
    if len(data) > MAX_INDEX_BYTES:
        raise ValueError(f"packed expert index is too large: {path}")
    return json.loads(data)


def read_pack_layout(directory, format_name=LAYOUT):
    # Only the file and tensor order are needed to compare bytes with the GGUF.
    if os.path.lexists(directory / INDEX):
        index = read_json(directory / INDEX)
        if index["version"] != 1 or index["layout"] != format_name:
            raise ValueError("unsupported packed expert index")
        return index["layers"]

    manifest = read_json(directory / MANIFEST)
    if (manifest["schema_version"], manifest["layout"], manifest["backend_layout"]) != (
            1, "layer_major_expert", format_name):
        raise ValueError("unsupported packed expert manifest")
    by_layer = {}
    for entry in manifest["entries"]:
        by_layer.setdefault(entry["layer"], []).append(entry)
    layout = []
    for layer, entries in sorted(by_layer.items()):
        entries.sort(key=lambda entry: entry["repacked_offset"])
        files = {entry["repacked_file"] for entry in entries}
        if len(files) != 1:
            raise ValueError(f"invalid packed files in layer {layer}")
        layout.append({"layer": layer, "file": files.pop(), "tensors": [e["tensor_name"] for e in entries]})
    return layout


def process_layer(source, packed, filename, names, tensors, pack, verify, stride):
    count = tensors[0].count
    inputs = [C.create_string_buffer(CHUNK_EXPERTS * tensor.raw_size) for tensor in tensors]
    record = C.create_string_buffer(stride)
    data = memoryview(record).cast("B")
    checksums = []
    output_offsets = []
    offset = 0
    for tensor in tensors:
        output_offsets.append(offset)
        offset += tensor.packed_size

    for first in range(0, count, CHUNK_EXPERTS):
        n = min(count - first, CHUNK_EXPERTS)
        for raw, name, tensor in zip(inputs, names, tensors):
            source.seek(tensor.offset + first * tensor.raw_size)
            size = n * tensor.raw_size
            if source.readinto(memoryview(raw)[:size]) != size:
                raise ValueError(f"short GGUF read: {name}")
        for i in range(n):
            for raw, name, tensor, offset in zip(inputs, names, tensors, output_offsets):
                if pack(C.byref(record, offset), C.byref(raw, i * tensor.raw_size),
                        tensor.kind, tensor.ne0, tensor.ne1) != tensor.packed_size:
                    raise ValueError(f"Hexagon packed size changed: {name}")
            if verify:
                if data != packed.read(stride):
                    raise ValueError(f"packed data mismatch: {filename}, expert {first + i}")
            else:
                packed.write(data)
            checksums.append(zlib.crc32(data))
    return checksums


def verify_stamp(path, stamp, file=None):
    if file is not None and file_stamp(os.fstat(file.fileno())) != stamp:
        raise ValueError(f"file changed while packing or verifying: {path}")
    if file_stamp(path.stat(follow_symlinks=False)) != stamp:
        raise ValueError(f"file changed while packing or verifying: {path}")


def process(source, model, source_stamp, directory, layers, pack, verify, format_name=LAYOUT):
    layout = read_pack_layout(directory, format_name) if verify else [
        {"layer": layer, "file": f"layer_{layer:03d}.bin", "tensors": sorted(tensors)}
        for layer, tensors in sorted(layers.items())
    ]
    if not layout:
        raise ValueError("packed expert index has no layers")

    filenames = {}
    layer_ids = set()
    index = []
    for row in layout:
        layer, name, names = row["layer"], row["file"], row["tensors"]
        if type(layer) is not int or not 0 <= layer <= 0x7fffffff:
            raise ValueError(f"invalid packed layer ID: {layer!r}")
        if not isinstance(name, str) or name in ("", ".", "..", INDEX, MANIFEST) or any(c in name for c in "/\\:\0"):
            raise ValueError(f"invalid packed file name: {name!r}")
        if name in filenames or layer in layer_ids:
            raise ValueError(f"duplicate packed file or layer: {name}")
        layer_ids.add(layer)

        tensors = layers.get(layer, {})
        if not isinstance(names, list) or not names or len(names) != len(tensors) or set(names) != tensors.keys():
            raise ValueError(f"incompatible expert list in layer {layer}")
        ordered = [tensors[tensor] for tensor in names]
        payload = sum(tensor.packed_size for tensor in ordered)
        stride = payload + (-payload % DIRECT_ALIGNMENT)
        size = stride * ordered[0].count
        path = directory / name
        with (open_regular(path) if verify else path.open("xb")) as packed:
            if verify:
                stamp = file_stamp(os.fstat(packed.fileno()))
                if stamp[2] != size:
                    raise ValueError(f"unexpected packed file size: {name}")
            checksums = process_layer(source, packed, name, names, ordered, pack, verify, stride)
            if verify:
                verify_stamp(path, stamp, packed)
            else:
                packed.flush()
                os.fsync(packed.fileno())
                stamp = file_stamp(os.fstat(packed.fileno()))
        filenames[name] = stamp
        index.append({"layer": layer, "file": name, "tensors": names, "checksums": checksums})
        print(f"{'verified' if verify else 'packed'} {name}: {size} bytes", flush=True)  # noqa: NP100

    verify_stamp(model, source_stamp, source)
    for name, stamp in filenames.items():
        verify_stamp(directory / name, stamp)

    sealed = {"version": 1, "layout": format_name, "source": source_stamp, "layers": index}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".index-", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(sealed, file, separators=(",", ":"))
            file.write("\n")
            file.flush()
            if os.fstat(file.fileno()).st_size > MAX_INDEX_BYTES:
                raise ValueError("packed expert index is too large")
            os.fsync(file.fileno())
            os.replace(temporary, directory / INDEX)
        finally:
            temporary.unlink(missing_ok=True)


def publish(libc, source, destination):
    # A peer can create the destination after argument validation.
    rename = bind(libc, "renameat2", C.c_int, C.c_int, C.c_char_p, C.c_int, C.c_char_p, C.c_uint)
    at_fdcwd, rename_noreplace = -100, 1
    if rename(at_fdcwd, os.fsencode(source), at_fdcwd, os.fsencode(destination), rename_noreplace):
        error = C.get_errno()
        raise OSError(error, os.strerror(error), destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="complete, single-file GGUF")
    parser.add_argument("--lib-dir", required=True, type=Path, help="directory with matching Hexagon libraries")
    parser.add_argument("--q3k-compact", action="store_true", help="store Q3_K experts with compact scale metadata")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out-dir", type=Path, help="new directory for packed experts")
    action.add_argument("--verify", type=Path, metavar="DIR", help="byte-compare and seal an existing pack against the GGUF")
    args = parser.parse_args()

    model = args.model.expanduser().resolve(strict=True)
    libs = args.lib_dir.expanduser().resolve(strict=True)
    libc = C.CDLL(None, use_errno=True)
    base = C.CDLL(str(libs / "libggml-base.so"), mode=C.RTLD_GLOBAL)
    hexagon = C.CDLL(str(libs / "libggml-hexagon.so"))
    pack_symbol = "ggml_backend_hexagon_pack_moe_expert"
    format_name = COMPACT_LAYOUT if args.q3k_compact else LAYOUT
    if args.q3k_compact:
        pack_symbol += "_compact"
    pack = bind(hexagon, pack_symbol, C.c_size_t,
                C.c_void_p, C.c_void_p, C.c_int, C.c_int64, C.c_int64)
    with open_regular(model) as source:
        stamp = file_stamp(os.fstat(source.fileno()))
        layers = read_model(base, libc, source, stamp[2], pack)
        if args.verify:
            directory = args.verify.expanduser().resolve(strict=True)
            process(source, model, stamp, directory, layers, pack, verify=True, format_name=format_name)
        else:
            destination = args.out_dir.expanduser().absolute()
            if os.path.lexists(destination):
                raise ValueError(f"output directory already exists: {destination}")
            with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f".{destination.name}-") as stage:
                process(source, model, stamp, Path(stage), layers, pack, verify=False, format_name=format_name)
                publish(libc, Path(stage), destination)


if __name__ == "__main__":
    try:
        main()
    except (AttributeError, KeyError, OSError, TypeError, ValueError) as exc:
        sys.exit(f"{Path(__file__).name}: {exc}")
