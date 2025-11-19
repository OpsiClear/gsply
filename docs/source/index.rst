Welcome to gsply
=================

**Ultra-fast Gaussian Splatting PLY I/O Library**

gsply is a high-performance Python library for reading and writing Gaussian Splatting PLY files.
Built with NumPy and Numba, it achieves **93M Gaussians/sec read** and **57M Gaussians/sec write**
performance through zero-copy optimizations and JIT-compiled compression pipelines.
Features **11x faster GPU transfers** using zero-copy `_base` tensor optimization and **automatic format
state tracking** via `_format` dictionary for seamless in-place conversions between PLY/linear formats
and SH/RGB color spaces.

.. raw:: html

   <div class="admonition note" style="margin-top: 1rem; margin-bottom: 2rem;">
   <p class="admonition-title">Quick Start</p>
   <p>
   <code>pip install gsply</code> |
   <code>from gsply import plyread, plywrite</code> |
   <code>data = plyread("model.ply")</code>
   </p>
   </div>

Key Features
------------

* **Zero-copy reads** — All data loaded as memory-efficient views into shared buffers
* **Auto-optimized writes** — Automatic consolidation for 2.4x faster writes
* **Dual format support** — Uncompressed PLY and PlayCanvas compressed (71-74% smaller)
* **Optimal GPU transfer** — 11x faster GPU transfers using zero-copy `_base` tensor (single transfer vs CPU stacking)
* **In-place conversion tracking** — Format state tracked automatically (`_format` dict) for scales, opacities, and colors
* **GPU acceleration** — Optional PyTorch integration with seamless CPU↔GPU transfers
* **Mask management** — Multi-layer boolean masks with GPU-optimized operations
* **Pure Python** — No C++ compilation required, works everywhere Python runs

Performance Highlights
-----------------------

* **Read**: 93M Gaussians/sec peak (400K Gaussians in ~6ms)
* **Write**: 57M Gaussians/sec peak (400K Gaussians in ~7ms zero-copy, ~19ms standard)
* **Compression**: 71-74% size reduction with PlayCanvas format
* **GPU transfer**: 11x faster with zero-copy `_base` optimization (1.99ms vs 22.78ms for 400K Gaussians)
* **Format conversion**: In-place operations track format state automatically (PLY ↔ linear, SH ↔ RGB)

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   overview
   usage
   changelog

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

Additional Resources
--------------------

* :ref:`genindex` — Complete function and class index
* :ref:`modindex` — Module index
* :ref:`search` — Full-text search

.. raw:: html

   <hr style="margin-top: 3rem; border: none; border-top: 1px solid #e0e0e0;">
   <p style="text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;">
   Documentation built with <a href="https://www.sphinx-doc.org/">Sphinx</a> and
   <a href="https://pradyunsg.me/furo/">Furo</a> theme.
   </p>
