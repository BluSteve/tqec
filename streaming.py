import csv
import random
from time import time
from typing import List

from scipy.constants import gibi

from tqec import BlockGraph
from tqec.circuit.qubit import GridQubit
from tqec.circuit.qubit_map import QubitMap
from tqec.compile.compile import compile_block_graph
from tqec.compile.tree.tree import LayerTree
from tqec.utils import TQECError
from tqec.utils.position import Position3D


def generate_qubit_map(x, y, k):
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


def memory(nx: int, ny: int, t: int) -> LayerTree:
    graph = BlockGraph()

    # cd = ['ZXX', 'ZXZ', 'XZX', 'XZZ']
    cd = ["ZXX", "ZXZ"]

    r = random.Random(100)

    graphs: List[BlockGraph | None] = [None] * t

    graphs[0] = BlockGraph()
    # Create n×n grid at each time slice
    for x in range(nx):
        for y in range(ny):
            graph.add_cube(Position3D(x, y, 0), r.choice(cd))
            graphs[0].add_cube(Position3D(x, y, 0), r.choice(cd))


    # Add spatial connections within the first layer
    for x in range(nx):
        for y in range(ny):
            try:
                if x < nx - 1:
                    graph.add_pipe(Position3D(x, y, 0), Position3D(x + 1, y, 0))
                    graphs[0].add_pipe(Position3D(x, y, 0), Position3D(x + 1, y, 0))
            except TQECError:
                pass

            try:
                if y < ny - 1:
                    graph.add_pipe(Position3D(x, y, 0), Position3D(x, y + 1, 0))
                    graphs[0].add_pipe(Position3D(x, y, 0), Position3D(x, y + 1, 0))
            except TQECError:
                pass

    # Add temporal layers and connections
    for i in range(1, t):
        gi = graphs[0] if i < 5 else BlockGraph()

        # Add all cubes in the n×n grid at time slice i
        for x in range(nx):
            for y in range(ny):
                graph.add_cube(Position3D(x, y, i), r.choice(cd))
                gi.add_cube(Position3D(x, y, i), r.choice(cd))
                # Add temporal pipe from previous time slice
                try:
                    graph.add_pipe(Position3D(x, y, i - 1), Position3D(x, y, i))
                    gi.add_pipe(Position3D(x, y, i - 1), Position3D(x, y, i))
                except TQECError:
                    pass

        # Add spatial connections within this layer
        if i != t - 1:
            for x in range(nx):
                for y in range(ny):
                    try:
                        if x < nx - 1:
                            graph.add_pipe(Position3D(x, y, i), Position3D(x + 1, y, i))
                            gi.add_pipe(Position3D(x, y, i), Position3D(x + 1, y, i))
                    except TQECError:
                        pass

                    try:
                        if y < ny - 1:
                            graph.add_pipe(Position3D(x, y, i), Position3D(x, y + 1, i))
                            gi.add_pipe(Position3D(x, y, i), Position3D(x, y + 1, i))
                    except TQECError:
                        pass

    # graph.view_as_html(f"memory_nx{nx}_ny{ny}_t{t}.html")

    graphs[0].view_as_html(f'memory_nx{nx}_ny{ny}_t{t}_layer0.html')

    compiled_graph = compile_block_graph(graph, observables="auto")

    return compiled_graph.to_layer_tree(), compile_block_graph(graphs[0]).to_layer_tree()


def benchmark():
    with open("results.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["n", "k", "duration1", "duration2"])

        for x in range(3, 20):
            start = time()

            lt = memory(x, 3)

            end = time()

            duration0 = end - start

            print(f"{x} {duration0}")

            # for k in range(7, 8):
            #     citer = lt.generate_circuit_stream(k)
            #     master_circuit = stim.Circuit()
            #
            #     start = time()
            #
            #     next(citer)  # annotations generation finish here
            #
            #     end = time()
            #
            #     duration1 = end - start
            #
            #     start = time()
            #
            #     i = 0
            #     for circ in citer:
            #         master_circuit += circ
            #         # print(circ)
            #         # print('----------------------------')
            #         i += 1
            #         # if i == 10:
            #         #     break
            #
            #     # with open(f"master_circuit_n{x}_k{k}.stim", "w") as f:
            #     #     f.write(str(master_circuit))
            #
            #     end = time()
            #     duration2 = end - start
            #     # writer.writerow([x, k, duration1, duration2])
            #
            #     print(f"{x} {k} {duration0} {duration1} {duration2}")


if __name__ == "__main__":
    nx = 3
    ny = 3
    t = 10
    k = 1

    # lt2 = memory(nx,ny, t)
    #
    # start = time()
    # circuit = lt2.generate_circuit(k)
    # end = time()
    #
    # print(f"Single circuit generation time: {end - start}\n\n")
    #
    # with open("circuit.txt", "w") as f:
    #     f.write(str(circuit))
    #
    # print('starting for real\n\n')

    magic_qm = generate_qubit_map(nx, ny, k)

    start = time()
    lt, lt0 = memory(nx, ny, t)
    end = time()
    print(f"Layer tree generation time: {end - start}\n\n")

    start = time()
    citer = lt0.generate_circuit_stream(k, magic_qm)
    end = time()
    print(f"Streamed annotations generation time: {end - start}\n\n")

    start = time()
    i = 0

    last = time()
    with open("master_circuit.txt", "w+") as f:
        for circ in citer:
            print(f"{i}, {time() - last}, {circ.__str__()[:20].replace('\n', ' ')}")
            last = time()
            i += 1
            f.write(str(circ) + "\n")

    end = time()
    print(f"Streamed circuit generation time: {end - start}\n\n")

    # print(circuit == master_circuit)

    print(lt)
