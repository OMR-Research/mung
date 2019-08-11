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
import argparse
import collections
import logging
import pprint
import time

import operator

from typing import List, Tuple, Dict, Any

from mung.io import read_nodes_from_file, get_edges
from mung.node import merge_node_lists_from_multiple_documents, Node


def compute_node_statistics(nodes: List[Node], edges: List[Tuple[int, int]] = None) -> Dict[str, Any]:
    stats = collections.OrderedDict()

    # Count Nodes
    stats['number_of_nodes'] = len(nodes)

    # Count Nodes by class
    number_of_nodes_by_class = collections.defaultdict(int)
    for node in nodes:
        number_of_nodes_by_class[node.class_name] += 1
    stats['number_of_nodes_by_class'] = number_of_nodes_by_class
    stats['number_of_distinct_nodes'] = len(number_of_nodes_by_class)

    if edges is not None:
        # Count relationships
        id_to_node_mapping = {node.id: node for node in nodes}
        stats['number_of_relationships'] = len(edges)
        number_of_relationships_by_class = collections.defaultdict(int)
        for edge in edges:
            from_node_id, to_node_id = edge
            from_node = id_to_node_mapping[from_node_id].class_name
            to_node = id_to_node_mapping[to_node_id].class_name
            number_of_relationships_by_class[(from_node, to_node)] += 1
        stats['number_of_relationships_by_class'] = number_of_relationships_by_class
        stats['number_of_relationships_distinct'] = len(number_of_relationships_by_class)

    return stats


def print_statistics(stats: Dict[str, Any]):
    # For now, just pretty-print. That means reformatting the insides
    # of the stats.
    print_stats = list()

    if 'number_of_nodes' in stats:
        print_stats.append(('number_of_nodes', stats['number_of_nodes']))
    if 'number_of_nodes_by_class' in stats:
        print_stats.append(('number_of_nodes_by_class',
                            sorted(list(stats['number_of_nodes_by_class'].items()),
                                   key=operator.itemgetter(1),
                                   reverse=True)
                            ))
    if 'number_of_distinct_nodes' in stats:
        print_stats.append(('number_of_distinct_nodes',
                            stats['number_of_distinct_nodes']))

    if 'number_of_relationships' in stats:
        print_stats.append(('number_of_relationships', stats['number_of_relationships']))
    if 'number_of_relationships_by_class' in stats:
        print_stats.append(('number_of_relationships_by_class',
                            sorted(list(stats['number_of_relationships_by_class'].items()),
                                   key=operator.itemgetter(1),
                                   reverse=True)))
    if 'number_of_relationships_distinct' in stats:
        print_stats.append(('number_of_relationships_distinct',
                            stats['number_of_relationships_distinct']))

    pprint.pprint(print_stats)


##############################################################################

def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', action='store', nargs='+',
                        required=True,
                        help='List of input NodeList files.')
    parser.add_argument('-e', '--emit', action='store', default='print',
                        choices=['print'],
                        help='How should the analysis results be presented?')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    logging.info('Starting main...')
    _start_time = time.clock()

    # Parse individual Node lists.
    node_lists = []
    number_of_parsed_nodes = 0
    for i, f in enumerate(args.input):
        node_list = read_nodes_from_file(f)
        node_lists.append(node_list)

        # Logging progress
        number_of_parsed_nodes += len(node_list)
        if i % 10 == 0 and i > 0:
            _time_parsing = time.clock() - _start_time
            nodes_per_second = number_of_parsed_nodes / _time_parsing
            logging.info('Parsed {0} Nodes in {1:.2f} s ({2:.2f} objs/s)'
                         ''.format(number_of_parsed_nodes,
                                   _time_parsing, nodes_per_second))

    # Merge the Node lists into one.
    # This is done so that the resulting object graph can be manipulated
    # at once, without id clashes.
    merged_node_list = merge_node_lists_from_multiple_documents(node_lists)

    edges = get_edges(merged_node_list)

    _parse_end_time = time.clock()
    logging.info('Parsing took {0:.2f} s'.format(_parse_end_time - _start_time))

    ##########################################################################
    # Analysis

    # Here's where the results are stored, for export into various
    # formats. (Currently, we only print them.)
    statistics = compute_node_statistics(merged_node_list, edges=edges)

    ##########################################################################
    # Export
    if args.emit == 'print':
        print_statistics(statistics)

    _end_time = time.clock()
    logging.info('analyze_annotations.py done in {0:.3f} s'
                 ''.format(_end_time - _start_time))

