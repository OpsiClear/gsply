API Reference
=============

Complete API documentation for all public functions, classes, and modules.
All public symbols are re-exported through the top-level `gsply` package,
so you can `import gsply` and access everything from a single namespace.

Core Modules
------------

.. toctree::
   :maxdepth: 2
   :caption: Core I/O

   reader
   writer
   formats

.. toctree::
   :maxdepth: 2
   :caption: Data Containers

   gsdata

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   utils

GPU Acceleration
----------------

.. toctree::
   :maxdepth: 2
   :caption: PyTorch Integration

   torch

Quick Reference
---------------

**Reading & Writing**

- :func:`gsply.plyread` — Read PLY files (auto-detects format)
- :func:`gsply.plywrite` — Write PLY files (auto-optimized)
- :func:`gsply.compress_to_bytes` — Compress to bytes
- :func:`gsply.decompress_from_bytes` — Decompress from bytes

**Data Containers**

- :class:`gsply.GSData` — CPU NumPy container
- :class:`gsply.GSTensor` — GPU PyTorch container

**Format Detection**

- :func:`gsply.detect_format` — Detect PLY format and SH degree
