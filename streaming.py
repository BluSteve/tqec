import csv
import random
from time import time

import stim

from tqec import BlockGraph
from tqec.compile.compile import compile_block_graph
from tqec.compile.tree.node import AnnotateDetectorsOnLayerNode
from tqec.compile.tree.tree import LayerTree
from tqec.utils import TQECError
from tqec.utils.position import Position3D


def memory(n: int, t: int) -> LayerTree:
    graph = BlockGraph()

    # cd = ['ZXX', 'ZXZ', 'XZX', 'XZZ']
    cd = ["ZXX", "ZXZ"]

    r = random.Random(100)

    # Create n×n grid at each time slice
    for x in range(n):
        for y in range(n):
            graph.add_cube(Position3D(x, y, 0), r.choice(cd))

    # Add spatial connections within the first layer
    for x in range(n):
        for y in range(n):
            try:
                if x < n - 1:
                    graph.add_pipe(Position3D(x, y, 0), Position3D(x + 1, y, 0))
            except TQECError:
                pass

            try:
                if y < n - 1:
                    graph.add_pipe(Position3D(x, y, 0), Position3D(x, y + 1, 0))
            except TQECError:
                pass

    # Add temporal layers and connections
    for i in range(1, t):
        # Add all cubes in the n×n grid at time slice i
        for x in range(n):
            for y in range(n):
                graph.add_cube(Position3D(x, y, i), r.choice(cd))
                # Add temporal pipe from previous time slice
                try:
                    graph.add_pipe(Position3D(x, y, i - 1), Position3D(x, y, i))
                except TQECError:
                    pass

        # Add spatial connections within this layer
        if i != t - 1:
            for x in range(n):
                for y in range(n):
                    try:
                        if x < n - 1:
                            graph.add_pipe(Position3D(x, y, i), Position3D(x + 1, y, i))
                    except TQECError:
                        pass

                    try:
                        if y < n - 1:
                            graph.add_pipe(Position3D(x, y, i), Position3D(x, y + 1, i))
                    except TQECError:
                        pass

    # graph.view_as_html(f"memory_n{n}_t{t}.html")

    compiled_graph = compile_block_graph(graph, observables="auto")

    return compiled_graph.to_layer_tree()


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
    lt2 = memory(5, 5)

    k = 2
    circuit = lt2.generate_circuit(k)

    print('starting for real\n\n')

    magic_qm = lt2._get_global_qubit_map(k)

    with open("circuit.txt", "w") as f:
        f.write(str(circuit))

    lt = memory(5, 5)
    citer = lt.generate_circuit_stream(k, qubit_map=magic_qm)

    master_circuit = stim.Circuit()
    i = 0
    for circ in citer:
        # print(i)
        i += 1
        master_circuit += circ

    with open("master_circuit.txt", "w") as f:
        f.write(str(master_circuit))

    print(circuit == master_circuit)

    print(lt)
