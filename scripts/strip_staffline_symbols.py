#!/usr/bin/env python
"""This is a script that removes all staffline, staffspace and staff
symbols, and all relationships that lead to them."""
import argparse
import copy
import logging
import os
import time

from mung.io import read_nodes_from_file, export_node_list

STAFF_CLSNAMES = ['staff', 'staff_line', 'staff_space']


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-a', '--annot', action='store', required=True,
                        help='The annotation file for which the staffline and staff'
                             ' CropObject relationships should be added.')
    parser.add_argument('-e', '--export', action='store',
                        help='A filename to which the output CropObjectList'
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
    logging.info('Import the CropObject list')
    if not os.path.isfile(args.annot):
        raise ValueError('Annotation file {0} not found!'
                         ''.format(args.annot))
    cropobjects = read_nodes_from_file(args.annot)

    ##########################################################################
    staff_cropobjects_dict = {c.objid: c for c in cropobjects
                              if c.clsname in STAFF_CLSNAMES}

    output_cropobjects = []
    for c in cropobjects:
        if c.objid in staff_cropobjects_dict:
            continue
        new_c = copy.deepcopy(c)
        new_c.inlinks = [i for i in c.inlinks
                         if i not in staff_cropobjects_dict]
        new_c.outlinks = [o for o in c.outlinks
                          if o not in staff_cropobjects_dict]
        output_cropobjects.append(new_c)

    ##########################################################################
    logging.info('Export the stripped list.')
    cropobject_string = export_node_list(output_cropobjects)

    if args.export is not None:
        with open(args.export, 'w') as hdl:
            hdl.write(cropobject_string)
    else:
        print(cropobject_string)

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
