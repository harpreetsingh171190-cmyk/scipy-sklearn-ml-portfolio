"""Compatibility shim.

The original 2017 course shipped a copy of scikit-learn's private tree exporter here.
scikit-learn now provides ``sklearn.tree.plot_tree`` natively, so we simply re-export it.
"""
from sklearn.tree import plot_tree

__all__ = ["plot_tree"]
