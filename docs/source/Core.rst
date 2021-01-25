Core
====
This section details the currently implemented base classes available in CAMP, including the Structured Grid and
Triangle Mesh objects. The structured grids represent images, fields, look-up tables, or anything that is structured on
a grid. The triangle mesh object inherits from an underlying unstructured grid object meant to represent different
surfaces. Currently, the only mesh type supported is a triangle mesh, but the unstructured grid object could easily
be expanded to include other mesh types, such as quads. This package also provides functions for displaying both
structured grid data and triangle mesh objects.

Structured Grid
---------------

.. automodule:: camp.Core.StructuredGridClass
   :members:
   :undoc-members:

Triangle Mesh
-------------

.. automodule:: camp.Core.TriangleMeshClass
   :members:
   :undoc-members:

Display
-------

.. automodule:: camp.Core.Display
   :members:
   :undoc-members:
