#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Codecs and filters registered from Python need the GIL, so C-Blosc2 must run them on the calling
# thread whatever the number of threads: BLOSC_NTHREADS replaces the nthreads of every context,
# super-chunks made from frames get the global number of threads and opened ones blosc2.nthreads.
# On failure the callbacks run on C-Blosc2's worker threads and crash the interpreter, so every
# case runs in a child interpreter.

import os
import subprocess
import sys
import textwrap

import pytest

import blosc2

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="subprocess is not supported on emscripten/wasm32")

CHILD = textwrap.dedent(
    """
    import sys

    import numpy as np

    import blosc2

    kind, source, threads_from, urlpath = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] or None
    ID = 170  # of the codec or the filter, in the range for user ids


    def encoder(input, output, meta, schunk):
        # 2 bytes per item: the values are below 2 ** 16
        n = input.size // 4
        output[: 2 * n] = input.view(np.int32).astype(np.uint16).view(np.uint8)
        return 2 * n


    def decoder(input, output, meta, schunk):
        n = output.size // 4
        output[:] = input[: 2 * n].view(np.uint16).astype(np.int32).view(np.uint8)
        return output.size


    def reverse(input, output, meta, schunk):
        output[:] = input[::-1]


    if kind == "codec":
        blosc2.register_codec("calling_thread_codec", ID, encoder, decoder)
        cparams = {"codec": ID, "filters": [blosc2.Filter.NOFILTER], "filters_meta": [0]}
    else:
        blosc2.register_filter(ID, reverse, reverse, "calling_thread_filter")
        cparams = {"codec": blosc2.Codec.LZ4, "filters": [ID], "filters_meta": [0]}
    # 16 blocks per chunk, so that C-Blosc2 would use its worker threads
    N = 16 * 1000
    cparams.update(typesize=4, nthreads=1, blocksize=4000, splitmode=blosc2.SplitMode.NEVER_SPLIT)
    values = (np.arange(3 * N) % 1000).astype(np.int32)
    s = blosc2.SChunk(chunksize=4 * N, data=values, cparams=cparams, dparams={"nthreads": 1}, urlpath=urlpath)
    if threads_from == "set_nthreads":
        blosc2.set_nthreads(4)
    if source == "frame":
        s = blosc2.schunk_from_cframe(s.to_cframe(), copy=True)
    elif source == "file":
        s = blosc2.open(urlpath, mode="a")
    out = np.empty_like(values)
    s.get_slice(out=out)
    assert np.array_equal(out, values)
    assert np.array_equal(np.frombuffer(s.decompress_chunk(1), np.int32), values[N : 2 * N])
    s.append_data(values[:N])
    assert np.array_equal(np.frombuffer(s.decompress_chunk(3), np.int32), values[:N])
    print("ok")
    """
)


@pytest.mark.parametrize("kind", ["codec", "filter"])
@pytest.mark.parametrize(
    ("source", "threads_from"),
    [
        # C-Blosc2 gives every context BLOSC_NTHREADS threads, also the ones asking for 1
        ("new", "BLOSC_NTHREADS"),
        # A super-chunk made from a frame gets C-Blosc2's global number of threads, an opened one
        # blosc2.nthreads
        ("frame", "set_nthreads"),
        ("file", "set_nthreads"),
    ],
)
def test_python_callbacks_calling_thread(kind, source, threads_from, tmp_path):
    env = dict(os.environ)
    env.pop("BLOSC_NTHREADS", None)
    if threads_from == "BLOSC_NTHREADS":
        env["BLOSC_NTHREADS"] = "4"
    urlpath = str(tmp_path / "calling_thread.b2frame") if source == "file" else ""
    result = subprocess.run(
        [sys.executable, "-c", CHILD, kind, source, threads_from, urlpath],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert result.stdout.split() == ["ok"]
