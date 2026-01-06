import os
import random
from time import time
from typing import Iterator, Tuple

import stim

from tqec import BlockGraph
from tqec.circuit.qubit import GridQubit
from tqec.circuit.qubit_map import QubitMap
from tqec.compile.compile import compile_block_graph
from tqec.computation.cube import ZXCube
from tqec.utils import TQECError
from tqec.utils.position import Position3D


def _generate_qubit_map(x, y, k) -> QubitMap:
    d = 2 * k + 1
    a = 2 * d

    xa = a * x + 2 * (x - 1) + 1
    ya = a * y + 2 * (y - 1) + 1

    qubits: dict[int, GridQubit] = {}

    i = 0
    for b in range(xa):
        for c in range(ya):
            if (b + c) % 2 == 1:
                continue
            qubits[i] = GridQubit(b, c)
            i += 1

    qm = QubitMap(qubits)

    return qm


def _random_block_graph(nx: int, ny: int, t: int) -> BlockGraph:
    if t < 3:
        raise TQECError("t must be at least 3")

    graph = BlockGraph()

    cd = ["ZXX", "ZXZ"]

    r = random.Random(100)

    # Create nxn grid at each time slice
    for x in range(nx):
        for y in range(ny):
            graph.add_cube(Position3D(x, y, 0), "P", label=f"input_{x}_{y}")

    # Add spatial connections within the first layer
    for x in range(nx):
        for y in range(ny):
            try:
                if x < nx - 1:
                    graph.add_pipe(Position3D(x, y, 0), Position3D(x + 1, y, 0))
            except TQECError:
                pass

            try:
                if y < ny - 1:
                    graph.add_pipe(Position3D(x, y, 0), Position3D(x, y + 1, 0))
            except TQECError:
                pass

    # Add temporal layers and connections
    for i in range(1, t - 1):
        # Add all cubes in the nxn grid at time slice i
        for x in range(nx):
            for y in range(ny):
                graph.add_cube(Position3D(x, y, i), r.choice(cd))
                # Add temporal pipe from previous time slice
                try:
                    graph.add_pipe(Position3D(x, y, i - 1), Position3D(x, y, i))
                except TQECError:
                    pass

        # Add spatial connections within this layer
        for x in range(nx):
            for y in range(ny):
                try:
                    if x < nx - 1:
                        graph.add_pipe(Position3D(x, y, i), Position3D(x + 1, y, i))
                except TQECError:
                    pass

                try:
                    if y < ny - 1:
                        graph.add_pipe(Position3D(x, y, i), Position3D(x, y + 1, i))
                except TQECError:
                    pass

    # Add ports to the last layer
    for x in range(nx):
        for y in range(ny):
            graph.add_cube(Position3D(x, y, t - 1), "P", label=f"output_{x}_{y}")

            try:
                graph.add_pipe(Position3D(x, y, t - 2), Position3D(x, y, t - 1))
            except TQECError:
                pass

    # fill ports randomly
    mapping = {}
    for x in range(nx):
        for y in range(ny):
            mapping[f"input_{x}_{y}"] = mapping[f"output_{x}_{y}"] = ZXCube.from_str(r.choice(cd))

    graph.fill_ports(mapping)

    return graph


def _partition_block_graph(block_graph: BlockGraph, z_per_partition: int) -> list[BlockGraph]:
    """Partition a block graph into smaller block graphs.

    Args:
        block_graph: The original block graph to partition.
        max_cubes_per_partition: Maximum number of cubes allowed in each partition.

    Returns:
        A list of partitioned block graphs.

    """
    if z_per_partition < 2:
        raise TQECError("z_per_partition must be at least 2")

    partitions: list[BlockGraph] = []

    # filter blockgraph by z
    zlo = 0
    zhi = z_per_partition - 1

    max_z = block_graph.bounding_box_size()[2]
    while zlo < max_z:
        cubes = set(filter(lambda cube: zlo <= cube.position.z <= zhi, block_graph.cubes))
        pipes = list(filter(lambda pipe: pipe.u in cubes and pipe.v in cubes, block_graph.pipes))
        graph_part = BlockGraph()
        for cube in cubes:
            graph_part.add_cube(cube.position, cube.kind)

        for pipe in pipes:
            graph_part.add_pipe(pipe.u.position, pipe.v.position, pipe.kind)

        partitions.append(graph_part)

        zlo += z_per_partition - 1
        zhi += z_per_partition - 1  # overlap

    return partitions


def benchmark_stream(
        nx: int,
        ny: int,
        t: int,
        k: int,
        compare_to_unstreamed: bool = False,
        write_blockgraph_to_disk: bool = False,
        block_graph_streaming: bool = False,
        partition_length: int = 10,
        stop_after_first_instr: bool = False
) -> float | None:
    """Benchmark streaming generation of quantum circuits for a compiled block graph.

    Generates a block graph, compiles it into a layer tree, and produces circuit
    instructions via streaming. Optionally compares streamed results with unstreamed
    generation. Logs timing metrics for performance analysis.

    NOTE: The qubit map used for streaming is different from the one used for unstreamed,
    even though the resulting circuit is functionally the same. To prove the unstreamed
    and streamed circuits are exactly the same, we need to use the same qubit map for both.
    This means piping the qubit map from the unstreamed generation into the streamed generation.

    Args:
        nx: Number of blocks along the x-axis in the block graph.
        ny: Number of blocks along the y-axis in the block graph.
        t: Time parameter for generating the block graph.
        k: Number of layers to use for circuit generation in the layer tree.
        compare_to_unstreamed: Whether to compare the streamed circuit with an
            unstreamed circuit. Defaults to False.

    """
    print(f"Benchmarking streaming with nx={nx}, ny={ny}, t={t}, k={k}\n")

    block_graph = _random_block_graph(nx, ny, t)

    if not block_graph_streaming:
        start = time()
        if write_blockgraph_to_disk:
            block_graph.view_as_html("block_graph.html")
            surfaces = block_graph.find_correlation_surfaces()
            print(f"Found {len(surfaces)} correlation surfaces.")
            for i, surface in enumerate(surfaces):
                block_graph.view_as_html(
                    f"block_graph_surface_{i}.html",
                    pop_faces_at_directions=("-Y", "+X"),
                    show_correlation_surface=surface,
                )

        compiled_graph = compile_block_graph(block_graph, observables="auto")
        lt = compiled_graph.to_layer_tree()
        end = time()
        print(f"Layer tree generation time (s): {end - start:.8f}\n")

    circuit = None
    if compare_to_unstreamed:
        lt2 = compiled_graph.to_layer_tree()  # a copy is needed here

        start = time()
        circuit = lt2.generate_circuit(k)
        end = time()
        print(f"Unstreamed circuit generation time (s): {end - start:.8f}\n")

        magic_qm = lt2._get_global_qubit_map(k)
    else:
        magic_qm = _generate_qubit_map(
            nx, ny, k
        )  # This qubit map is not tight on the qubits needed.


    init_time = time()

    start = time()
    citer = block_graph_stream(block_graph, k, magic_qm, partition_length) if block_graph_streaming else (
        lt.generate_circuit_stream(k, magic_qm))
    end = time()
    print(f"Initial stream generation time (s): {end - start:.8f}\n")

    start = time()

    last = time()
    master_circuit = stim.Circuit() if compare_to_unstreamed else None
    circuit_len = 0
    with open("master_circuit.txt", "w+") as f:
        print("Index, Time Taken (s), Time Elapsed (s), Total Circuit Size (bytes), Circuit Snippet")
        i = 0
        for circ in citer:
            if type(circ) == tuple:
                prx = circ[1]
                circ = circ[0]
            else:
                prx = circ

            f.write(str(circ))
            circuit_len += len(circ)
            print(f"{i}, {time() - last:.8f}, {time() - start:.8f}, {circuit_len}, " + circ.__str__()[:20].replace("\n", " "))
            if stop_after_first_instr and 'RX' in circ.__str__():
                print("Found RX in circuit!")
                return time() - init_time

            last = time()
            i += 1
            if compare_to_unstreamed:
                master_circuit += circ

    end = time()
    print(f"Total streamed circuit generation time (s): {end - start:.8f}\n")

    if compare_to_unstreamed:
        print("Comparing streamed vs unstreamed circuits...")
        same = master_circuit == circuit
        if same:
            print("Circuits are the same!")
        else:
            print("Circuits are different!")
        assert same


def block_graph_stream(graph: BlockGraph, k: int, qubit_map: QubitMap, partition_length: int) -> Iterator[Tuple[stim.Circuit, stim.Circuit | None]]:
    partitions = _partition_block_graph(graph, partition_length)

    j = 0
    for p in partitions:
        start = time()
        compiled_p = compile_block_graph(p, observables=None) # todo does not work with observables
        lt = compiled_p.to_layer_tree()
        end = time()
        print(f"\nPartition {j+1}/{len(partitions)} layer tree generation time (s): {end - start}")

        iter2 = lt.generate_circuit_stream(k, qubit_map)

        i = 0
        buf = None
        for x in iter2:
            i += 1
            if j != 0 and i < 6:
                continue
            if buf is not None:
                yield buf, x # yields x just for logging purposes
            buf = x

        # this does not include the last buffer
        if j == len(partitions) - 1:
            yield buf, None

        j += 1

def response_times():
    ns = [3, 5, 10, 30, 50]
    ks = [1, 2, 5, 10, 20]
    t = 5

    nst = dict()
    kst = dict()

    for n in ns:
        tt = benchmark_stream(n, n, t, 1, compare_to_unstreamed=False, write_blockgraph_to_disk=False,
                     block_graph_streaming=True, partition_length=2, stop_after_first_instr=True)
        print(f'n={n} took {tt :.8f} seconds')
        nst[n] = tt

    for k in ks:
        tt = benchmark_stream(3, 3, t, k, compare_to_unstreamed=False, write_blockgraph_to_disk=False,
                         block_graph_streaming=True, partition_length=2, stop_after_first_instr=True)
        print(f'k={k} took {tt :.8f} seconds')
        kst[k] = tt

    return nst, kst



if __name__ == "__main__":
    nst, kst = response_times()
    print(f'nst: {nst}')
    print(f'kst: {kst}')

    # nx = ny = 30
    # t = 10
    # k = 1
    #
    # graph = _random_block_graph(nx, ny, t)
    # # graph.view_as_html("block_graph.html")
    #
    # benchmark_stream(nx, ny, t, k, compare_to_unstreamed=False, write_blockgraph_to_disk=False,
    #                  block_graph_streaming=True, partition_length=2)

    # qmap = _generate_qubit_map(nx, ny, k)
    #
    # iter1 = (
    #     compile_block_graph(graph, observables='auto')
    #     .to_layer_tree()
    #     .generate_circuit_stream(k, qmap)
    # )
    #
    # stim1 = stim.Circuit()
    # for x in iter1:
    #     stim1 += x
    #
    #     # output both circuits to files
    # with open("stim1.txt", "w+") as f:
    #     f.write(str(stim1))
    #
    # iter2 = block_graph_stream(graph, k, qmap)
    #
    # stim2 = stim.Circuit()
    # for x in iter2:
    #     stim2 += x
    #
    # with open("stim2.txt", "w+") as f:
    #     f.write(str(stim2))
    #
    # print(f'Stim circuits are the same: {stim1 == stim2}')


