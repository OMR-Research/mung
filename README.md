# MuNG

The ``mung`` (**Mu**sic **N**otation **G**raph) package implements a graph representation
 of music notation that is especially amenable to Optical Music Recognition (OMR).
It was used for instance in the [MUSCIMA++](https://ufal.mff.cuni.cz/muscima) dataset of music notation.

[![Build Status](https://travis-ci.org/OMR-Research/mung.svg?branch=master)](https://travis-ci.org/OMR-Research/mung)
[![PyPI version](https://badge.fury.io/py/mung.svg)](https://badge.fury.io/py/mung)
[![Documentation Status](https://readthedocs.org/projects/mung/badge/?version=latest)](https://mung.readthedocs.io/en/latest/?badge=latest)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE.txt)
[![codecov](https://codecov.io/gh/OMR-Research/mung/branch/master/graph/badge.svg)](https://codecov.io/gh/OMR-Research/mung)

Requires Python >= 3.6.


## Getting started

1. Install this package: ``pip install mung``
2. Download the [MUSCIMA++ dataset](https://github.com/OMR-Research/muscima-pp).
3. Run through the [tutorial](https://muscima.readthedocs.io/en/latest/Tutorial.html#tutorial).

Fundamentally, the Music Notatation Graph is a very simple construct:

![](doc/MuNG%20Class%20Diagram.png)

It stores the primitives that can be detected by a [Music Object Detector](https://github.com/apacha/MusicObjectDetector-TF)
as nodes and then store the relations between those nodes. But the devil is
in the details. To better understand what kind of [relations are useful](https://archives.ismir.net/ismir2019/paper/000006.pdf) 
and which kind of relations are stored for common western music notation, check out the
[annotator instruction from MUSCIMarker](https://muscimarker.readthedocs.io/en/latest/instructions.html).



## Dataset
The dataset itself is available for download
[here](https://github.com/OMR-Research/muscima-pp) and due to its derived nature, licensed differently:

[![abc](https://img.shields.io/badge/Dataset_Version-2.0-brightgreen.svg)](https://github.com/OMR-Research/muscima-pp)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


### Introduction to MUSCIMA++ Video

Watch Jan give a 30 minute introduction into this dataset on YouTube, which explains many design decisions and thoughts that directly affected the creation of the MuNG format:

[![Introduction to MUSCIMA++](https://img.youtube.com/vi/SvBvcxdGoE8/0.jpg)](https://www.youtube.com/watch?v=SvBvcxdGoE8)

