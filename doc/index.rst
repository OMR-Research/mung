.. muscima documentation master file, created by
   sphinx-quickstart on Sun Feb  5 20:12:02 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

muscima -- tools for the MUSCIMA++ dataset
==========================================

The ``muscima`` package implements tools for easier manipulation of the MUSCIMA++
dataset. Download the dataset here:

`<https://ufal.mff.cuni.cz/muscima/download>`_

A description of the dataset is on the project's homepage:

`<https://ufal.mff.cuni.cz/muscima>`_

And more thoroughly in an arXiv.org publication:

`<https://arxiv.org/pdf/1703.04824.pdf>`_

This pacakge is licensed under the MIT license (see ``LICENSE.txt`` file).
The package author is Jan Hajič jr. You can contact him at::

  hajicj@ufal.mff.cuni.cz

Questions and comments are welcome! This package is also hosted on github,
so if you find a bug, submit an issue (or a pull request!) there:

`<https://github.com/hajicj/muscima>`_


Requirements
------------

Python 3.5, otherwise nothing beyond the ``requirements.txt`` file: ``lxml`` and ``numpy``.
If you want to apply pitch inference, you should also get ``music21``.

Installation
------------

If you have ``pip``, just run::

  pip install muscima

If you don't have ``pip``, then you should `get it <https://pypi.python.org/pypi/pip>`_.
Or use the `Anaconda distribution <https://www.continuum.io/>`_.

First steps
-----------

Let's first download the dataset::

  curl https://ufal.mff.cuni.cz/~hajicj/2017/docs/MUSCIMA_0.9.zip > MUSCIMA++_0.9.zip
  unzip MUSCIMA++_0.9.zip
  cd MUSCIMA++_0.9

Take a look at the dataset's ``README.md`` file first. You can also read it online:

`<https://ufal.mff.cuni.cz/muscima>`_

Please make sure you understand the license requirements -- the data is licensed
as CC-BY-NC-SA 4.0, and because it is built over a previous dataset, there are *two*
attributions required.

Next, we fire up ``ipython`` (or just the plain ``python`` console, but definitely check out
ipython if you don't use it!) and parse the data::

  ipython
  >>> import os
  >>> from muscima.io import read_nodes_from_file
  >>> cropobject_fnames = [os.path.join('data', 'cropobjects', f) for f in os.listdir('data/cropobjects')]
  >>> docs = [read_nodes_from_file(f) for f in cropobject_fnames]
  >>> len(docs)
  140

In ``docs``, we now have a list of CropObject lists for each of the 140 documents.

Now that the dataset has been parsed, we can try to do some experiments!
We can do for example symbol classification. Go check out the :ref:`tutorial`!


Contents
--------

.. toctree::
   :maxdepth: 4
   muscima
   scripts

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

