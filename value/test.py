"""Interactive entrypoint for the scalar autograd demo and tests."""

from __future__ import annotations

import argparse
import unittest

from test_value import ValueBackpropTests
from value import Value


def build_demo_graph() -> dict[str, Value]:
    """Create a small graph that is easy to inspect interactively."""
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    prod = a * b
    shifted = prod + c
    out = shifted.tanh()
    return {
        "a": a,
        "b": b,
        "c": c,
        "prod": prod,
        "shifted": shifted,
        "out": out,
    }


def print_graph(graph: dict[str, Value]) -> None:
    for name, node in graph.items():
        print(f"{name}: data={node.data:.6f}, grad={node.grad:.6f}, op={node.op or 'input'}")


def run_demo() -> dict[str, Value]:
    graph = build_demo_graph()
    graph["out"].backward()
    print_graph(graph)
    return graph


def run_tests(verbosity: int = 2) -> unittest.result.TestResult:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValueBackpropTests)
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true", help="run the unittest suite")
    parser.add_argument("--demo", action="store_true", help="run the demo graph and print node values")
    args = parser.parse_args()

    if args.tests:
        run_tests()
    if args.demo or not args.tests:
        print("Demo graph after backward():")
        graph = run_demo()
        print("\nInteractive names available: Value, ValueBackpropTests, build_demo_graph, run_demo, run_tests, graph")
        globals()["graph"] = graph


if __name__ == "__main__":
    main()
