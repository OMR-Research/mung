"""This is a script that removes all staffline, staffspace and staff
symbols, and all relationships that lead to them."""
import argparse
import copy
import logging
import os
import time

from mung.io import read_nodes_from_file, export_node_list
from mung.constants import InferenceEngineConstants as _CONST


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-a', '--annot', action='store', required=True,
                        help='The annotation file for which the staffline and staff'
                             ' Node relationships should be added.')
    parser.add_argument('-e', '--export', action='store',
                        help='A filename to which the output NodeList'
                             ' should be saved. If not given, will print to'
                             ' stdout.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    # Your code goes here
    ##########################################################################
    logging.info('Import the Node list')
    if not os.path.isfile(args.annot):
        raise ValueError('Annotation file {0} not found!'
                         ''.format(args.annot))
    nodes = read_nodes_from_file(args.annot)

    ##########################################################################
    staff_id_to_node_mapping = {node.id: node for node in nodes
                              if node.class_name in _CONST.STAFF_CLASS_NAMES}

    output_nodes = []
    for node in nodes:
        if node.id in staff_id_to_node_mapping:
            continue
        new_c = copy.deepcopy(node)
        new_c.inlinks = [i for i in node.inlinks
                         if i not in staff_id_to_node_mapping]
        new_c.outlinks = [o for o in node.outlinks
                          if o not in staff_id_to_node_mapping]
        output_nodes.append(new_c)

    ##########################################################################
    logging.info('Export the stripped list.')
    nodes_string = export_node_list(output_nodes)

    if args.export is not None:
        with open(args.export, 'w') as hdl:
            hdl.write(nodes_string)
    else:
        print(nodes_string)

    _end_time = time.clock()
    logging.info('strip_staffline_symbols.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
