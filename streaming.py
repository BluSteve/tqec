import stim

from tqec import BlockGraph
from tqec.compile.compile import compile_block_graph
from tqec.compile.tree.tree import LayerTree
from tqec.utils.position import Position3D


def memory(n: int) -> LayerTree:
    graph = BlockGraph()
    graph.add_cube(Position3D(0, 0, 0), "XZZ")
    for i in range(1, n):
        graph.add_cube(Position3D(0, 0, i), "XZZ")
        graph.add_pipe(Position3D(0, 0, i - 1), Position3D(0, 0, i))

    compiled_graph = compile_block_graph(graph)

    return compiled_graph.to_layer_tree()


if __name__ == "__main__":
    lt = memory(1)

    citer = lt.generate_circuit(1)

    master_circuit = stim.Circuit()
    for circ in citer:
        print(circ)
        master_circuit += circ

    print(master_circuit)

    """
    for p in range(1, 4):
        n = 10**p
        start = time()
        layer_tree = memory(n)
        end = time()
        duration = end - start
        print(f"[{p=}] {duration:.2f}s")
    """
