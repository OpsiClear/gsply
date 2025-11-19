API Reference
=============

Complete API documentation for all public functions, classes, and modules.
All public symbols are re-exported through the top-level `gsply` package,
so you can `import gsply` and access everything from a single namespace.

Core Modules
------------

Core I/O
~~~~~~~~

Reader module
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   reader

Functions:
- :func:`gsply.plyread` — Read PLY files (auto-detects format)
- :func:`gsply.read_uncompressed` — Read uncompressed PLY files
- :func:`gsply.read_compressed` — Read compressed PLY files
- :func:`gsply.decompress_from_bytes` — Decompress from bytes

SOG Reader module
^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   sog_reader

Functions:
- :func:`gsply.sogread` — Read SOG format files (requires `gsply[sogs]`)

Writer module
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   writer

Functions:
- :func:`gsply.plywrite` — Write PLY files (auto-optimized)
- :func:`gsply.write_uncompressed` — Write uncompressed PLY files
- :func:`gsply.write_compressed` — Write compressed PLY files
- :func:`gsply.compress_to_bytes` — Compress to bytes
- :func:`gsply.compress_to_arrays` — Compress to arrays

Format helpers
^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   formats

Functions:
- :func:`gsply.detect_format` — Detect PLY format and SH degree
- :func:`gsply.get_sh_degree_from_property_count` — Get SH degree from property count

Data Containers
---------------

GSData container
~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   gsdata

Classes:
- :class:`gsply.GSData` — CPU NumPy container

Utilities
---------

Utility helpers
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   utils

Functions:
- :func:`gsply.sh2rgb` — Convert spherical harmonics to RGB colors
- :func:`gsply.rgb2sh` — Convert RGB colors to spherical harmonics
- :func:`gsply.sigmoid` — Compute sigmoid function
- :func:`gsply.logit` — Compute logit function

Constants:
- :data:`gsply.SH_C0` — Spherical harmonic DC coefficient normalization constant

GPU Acceleration
----------------

PyTorch Integration
~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   torch

GPU I/O Functions:
- :func:`gsply.plyread_gpu` — Read compressed PLY directly to GPU
- :func:`gsply.plywrite_gpu` — Write GSTensor to compressed PLY using GPU compression

GSTensor Data Container:
- :class:`gsply.GSTensor` — GPU PyTorch container

Quick Reference
---------------

**Reading & Writing**

- :func:`gsply.plyread` — Read PLY files (auto-detects format)
- :func:`gsply.plywrite` — Write PLY files (auto-optimized)
- :func:`gsply.sogread` — Read SOG format files (requires `gsply[sogs]`)
- :func:`gsply.plyread_gpu` — Read compressed PLY directly to GPU
- :func:`gsply.plywrite_gpu` — Write GSTensor to compressed PLY using GPU compression
- :func:`gsply.compress_to_bytes` — Compress to bytes
- :func:`gsply.decompress_from_bytes` — Decompress from bytes

**Data Containers**

- :class:`gsply.GSData` — CPU NumPy container
- :class:`gsply.GSTensor` — GPU PyTorch container

**Format Detection**

- :func:`gsply.detect_format` — Detect PLY format and SH degree
