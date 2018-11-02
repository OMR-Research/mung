#!/usr/bin/env python
"""``analyze_annotation.py`` is a script that analyzes annotation results.

For an overview of command-line options, call::

  analyze_annotation.py -h


Functionality
-------------

* Count symbols
* Count symbol classes
* Compute symbol parameters per class (size, morphological features..?) [NOT IMPLEMENTED]

* Count relationships
* Count relationship classes
* Compute relationship parameters per class pair

"""
from __future__ import print_function, unicode_literals
from __future__ import division
import argparse
import collections
import json
import logging
import pprint
import time

import operator

from mung.io import parse_cropobject_list, export_cropobject_graph
from mung.node import merge_cropobject_lists

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


def compute_cropobject_stats(cropobjects, edges=None):
    stats = collections.OrderedDict()

    # Count cropobjects
    stats['n_cropobjects'] = len(cropobjects)

    # Count cropobjects by class
    n_cropobjects_by_class = collections.defaultdict(int)
    for c in cropobjects:
        n_cropobjects_by_class[c.clsname] += 1
    stats['n_cropobjects_by_class'] = n_cropobjects_by_class
    stats['n_cropobjects_distinct'] = len(n_cropobjects_by_class)

    if edges is not None:
        # Count relationships
        _cropobjects_dict = {c.objid: c for c in cropobjects}
        stats['n_relationships'] = len(edges)
        n_relationships_by_class = collections.defaultdict(int)
        for e in edges:
            fr, to = e
            c_fr = _cropobjects_dict[fr].clsname
            c_to = _cropobjects_dict[to].clsname
            n_relationships_by_class[(c_fr, c_to)] += 1
        stats['n_relationships_by_class'] = n_relationships_by_class
        stats['n_relationships_distinct'] = len(n_relationships_by_class)

    return stats


def emit_stats_pprint(stats):
    # For now, just pretty-print. That means reformatting the insides
    # of the stats.
    print_stats = list()

    if 'n_cropobjects' in stats:
        print_stats.append(('n_cropobjects', stats['n_cropobjects']))
    if 'n_cropobjects_by_class' in stats:
        print_stats.append(('n_cropobjects_by_class',
                            sorted(list(stats['n_cropobjects_by_class'].items()),
                                   key=operator.itemgetter(1),
                                   reverse=True)
                            ))
    if 'n_cropobjects_distinct' in stats:
        print_stats.append(('n_cropobjects_distinct',
                            stats['n_cropobjects_distinct']))

    if 'n_relationships' in stats:
        print_stats.append(('n_relationships', stats['n_relationships']))
    if 'n_relationships_by_class' in stats:
        print_stats.append(('n_relationships_by_class',
                            sorted(list(stats['n_relationships_by_class'].items()),
                                   key=operator.itemgetter(1),
                                   reverse=True)))
    if 'n_relationships_distinct' in stats:
        print_stats.append(('n_relationships_distinct',
                            stats['n_relationships_distinct']))


    pprint.pprint(print_stats)


##############################################################################

def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', action='store', nargs='+',
                        required=True,
                        help='List of input CropObjectList files.')
    parser.add_argument('-e', '--emit', action='store', default='print',
                        choices=['print', 'latex', 'json'],
                        help='How should the analysis results be presented?')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    # Parse individual Node lists.
    cropobject_lists = []
    _n_parsed_cropobjects = 0
    for i, f in enumerate(args.input):
        cs = parse_cropobject_list(f)
        cropobject_lists.append(cs)

        # Logging progress
        _n_parsed_cropobjects += len(cs)
        if i % 10 == 0 and i > 0:
            _time_parsing = time.clock() - _start_time
            _cropobjects_per_second = _n_parsed_cropobjects / _time_parsing
            logging.info('Parsed {0} cropobjects in {1:.2f} s ({2:.2f} objs/s)'
                         ''.format(_n_parsed_cropobjects,
                                   _time_parsing, _cropobjects_per_second))

    # Merge the Node lists into one.
    # This is done so that the resulting object graph can be manipulated
    # at once, without objid clashes.
    cropobjects = merge_cropobject_lists(*cropobject_lists)

    edges = export_cropobject_graph(cropobjects)

    _parse_end_time = time.clock()
    logging.info('Parsing took {0:.2f} s'.format(_parse_end_time - _start_time))

    ##########################################################################
    # Analysis

    # Here's where the results are stored, for export into various
    # formats. (Currently, we only print them.)
    stats = compute_cropobject_stats(cropobjects, edges=edges)

    ##########################################################################
    # Export
    if args.emit == 'print':
        emit_stats_pprint(stats)
    # More export options:
    #  - json
    #  - latex table

    _end_time = time.clock()
    logging.info('analyze_annotations.py done in {0:.3f} s'
                 ''.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
