#!/usr/bin/env python
"""This is a script that converts a simple line-based list
of MLClass names into the MLClassList XML definition file
for OMR toolbox. A time-saving utility.

Accepts either one class name per line:

notehead
stem
flag
slur
tie
barline

or one class name and tab-separated class group per line:

notehead    note
stem    note
flag    note
slur    notation
tie     notation
barline layout

If no group is specified, the ``<Folder>`` tag will be identical
to the class name. If group is specified, the folder will be
``group/classname``.

Colors
------

Symbols in the same group get the same color.

If no group is given, the color changes along the matplotlib
color cycle.

"""
from __future__ import print_function, unicode_literals, division
from builtins import object
import argparse
import codecs
import collections
import colorsys
import logging
import os
import time

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."

header = u'''<?xml version="1.0" encoding="utf-8"?>
<MLClassList noNamespaceSchema="mff-muscima-mlclasses.xsd">
\t<MLClasses>'''

footer = u'''
\t</MLClasses>
</MLClassList>
'''


def rgb2hex(rgb):
    """Converts a triplet of (R, G, B) floats in 0-1 range
    into a hex rgb string."""
    int_rgb = [int(255 * x) for x in rgb]
    return u'#' + u''.join([u'{:02X}'.format(c) for c in int_rgb])


class MLClassGenerator(object):
    def __init__(self):
        self.ctr = 0

        self.joiner = u'\n\t\t\t'
        self.header = u'\t\t<NodeClass>'
        self.footer = u'</NodeClass>'

        self.default_color = u'#FF6060'

        self.color_RGB = (1.0, 0.4, 0.4)  # Used for output
        self.color_HSV = colorsys.rgb_to_hsv(*self.color_RGB)  # Used internally

        self.delta_hue = 0.017
        self.delta_hue_group = 0.37

        #: Each group has its own list of colors. Their starts
        #  are separated by self.delta_hue_group. The last member
        #  of each group_colordict value is the next color for that
        #  group.
        self.group_colordict = collections.OrderedDict()

    def next_color(self, group=None):

        if group is None:
            output = self.color_RGB

            next_hue = (self.color_HSV[0] + self.delta_hue) % 1.0
            next_sat = self.color_HSV[1]
            next_value = self.color_HSV[2]

            self.color_HSV = (next_hue, next_sat, next_value)
            self.color_RGB = colorsys.hsv_to_rgb(*self.color_HSV)
        else:
            if group not in self.group_colordict:
                if len(self.group_colordict) == 0:
                    self.group_colordict[group] = [self.color_HSV]
                else:
                    last_group_color = list(self.group_colordict.values())[-1][0]
                    this_group_color = ((last_group_color[0] + self.delta_hue_group) % 1.0,
                                        last_group_color[1],
                                        last_group_color[2])
                    self.group_colordict[group] = [this_group_color]

            current_color = self.group_colordict[group][-1]
            output = colorsys.hsv_to_rgb(*current_color)

            next_hue = (current_color[0] + self.delta_hue) % 1.0
            next_sat = current_color[1]
            next_value = current_color[2]

            next_color = (next_hue, next_sat, next_value)
            self.group_colordict[group].append(next_color)

            logging.debug('Group: {0}, current color: {1}, next color: {2}, '
                          'groupdict for g: {3}'
                          ''.format(group, current_color, next_color, self.group_colordict[group]))

        return output

    def next_mlclass(self, cname, group=None):

        lines = [self.header]

        lines.append(u'<Id>{0}</Id>'.format(self.ctr))
        self.ctr += 1

        lines.append(u'<Name>{0}</Name>'.format(cname))

        folder = u'{0}'.format(cname)
        if group is not None:
            folder = u'{0}'.format(os.path.join(group, cname))
        lines.append(u'<GroupName>{0}</GroupName>'.format(folder))

        color = self.next_color(group=group)
        hexcolor = rgb2hex(color)
        lines.append(u'<Color>{0}</Color>'.format(hexcolor))

        lines.append(self.footer)

        output = self.joiner.join(lines)
        return output


def parse_classnames(filename):
    """Generator for class name (plus possibly group)."""
    for line in codecs.open(filename, 'r', 'utf-8'):
        if len(line.strip()) == 0:
            continue
        fields = line.strip().split(u'\t')
        if len(fields) == 0:  # Skip empty lines
            continue
        if fields[0][0] == '#':  # Skip comments
            continue

        cname = fields[0]
        group = None
        if len(fields) > 1:
            group = fields[1]

        yield cname, group


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input', action='store',
                        required=True,
                        help='The input CSV file.')
    parser.add_argument('-o', '--output', action='store',
                        required=True,
                        help='The output MFF-MUSCIMA XML file.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    generator = MLClassGenerator()

    lines = [header]

    for classname, group in parse_classnames(args.input):
        lines.append(generator.next_mlclass(classname, group=group))

    lines.append(footer)
    with codecs.open(args.output, 'w', 'utf-8') as output_handle:
        output_handle.write(u'\n'.join(lines))

    _end_time = time.clock()
    logging.info('classes2mlclasslist.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
