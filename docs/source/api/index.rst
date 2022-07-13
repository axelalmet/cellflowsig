.. cellflowsig documentation master file, created by
   Axel Almet on Sat Jul 9 09:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: cellflowsig
.. automodule:: cellflowsig
   :noindex:

API
==================================

Preprocessing: pp
-------------------

.. module:: cellflowsig.pp
.. currentmodule:: cellflowsig

.. autosummary::
   :toctree: .

   pp.construct_base_networks
   pp.construct_celltype_ligand_expressions


Tools: tl
-----------

.. module:: cellflowsig.tl
.. currentmodule:: cellflowsig

Causal signal learning
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.learn_causal_network
   tl.validate_against_base_network


Plotting: pl
------------

.. module:: cellflowsig.pl
.. currentmodule:: cellflowsig

.. autosummary::
   :toctree: .