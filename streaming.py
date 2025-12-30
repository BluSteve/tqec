import random
from time import time

import stim

from tqec import BlockGraph
from tqec.compile.compile import compile_block_graph
from tqec.compile.tree.tree import LayerTree
from tqec.utils import TQECError
from tqec.utils.position import Position3D


def memory(n: int, t: int) -> LayerTree:
    graph = BlockGraph()

    c1 = "ZXX"
    c2 = "ZXZ"

    r = random.Random(100)

    # Create n×n grid at each time slice
    for x in range(n):
        for y in range(n):
            graph.add_cube(Position3D(x, y, 0), c1 if r.randint(0, 1) == 1 else c2)

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
                graph.add_cube(Position3D(x, y, i), c1 if r.randint(0, 1) == 1 else c2)
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

    graph.view_as_html("memory.html")

    compiled_graph = compile_block_graph(graph)

    return compiled_graph.to_layer_tree()


if __name__ == "__main__":
    lt = memory(3, 5)

    for k in range(1, 50, 1):
        citer = lt.generate_circuit(k)
        master_circuit = stim.Circuit()

        start = time()

        next(citer)  # annotations generation finish here

        end = time()

        duration1 = end - start

        start = time()

        i = 0
        for circ in citer:
            master_circuit += circ
            # print(circ)
            # print('----------------------------')
            i += 1
            if i == 10:
                break

        end = time()
        duration2 = end - start
        print(f"[{k=}] {duration1:.4f}s {duration2:.4f}s")
