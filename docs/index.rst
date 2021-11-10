.. BLADE documentation master file, created by
   sphinx-quickstart on Tue Nov  9 23:22:27 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BLADE's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

BLADE
=====

Blade is a CUDA accelerated library that provides DSP Modules for radio-telescopes
like the Allen Telescope Array. These Modules are used in conjunction with hashpipe
to create an real-time beamformer. A group of modules working together for a common
purpose is called a Pipeline.

.. doxygenclass:: Blade
   :content-only:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
