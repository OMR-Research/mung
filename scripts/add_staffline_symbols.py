#!/usr/bin/env python
"""The script ``add_staffline_symbols.py`` takes as input a CVC-MUSCIMA
(page, writer) index and a corresponding CropObjectList file
and adds to the CropObjectList staffline and staff objects."""
from __future__ import print_function, unicode_literals
from __future__ import division
from builtins import zip
from builtins import range
import argparse
import logging
import os
import time

import numpy

from skimage.io import imread
from skimage.measure import label
from skimage.morphology import watershed
from skimage.filters import gaussian
import matplotlib.pyplot as plt

from mung.dataset import CVC_MUSCIMA
from mung.io import parse_cropobject_list, export_cropobject_list
from mung.node import Node
from mung.utils import connected_components2bboxes, compute_connected_components

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


STAFFLINE_CLSNAME = 'staff_line'
STAFFSPACE_CLSNAME = 'staff_space'
STAFF_CLSNAME = 'staff'

#
# def connected_components2bboxes(labels):
#     """Returns a dictionary of bounding boxes (upper left c., lower right c.)
#     for each label.
#
#     >>> labels = [[0, 0, 1, 1], [2, 0, 0, 1], [2, 0, 0, 0], [0, 0, 3, 3]]
#     >>> bboxes = connected_components2bboxes(labels)
#     >>> bboxes[0]
#     [0, 0, 4, 4]
#     >>> bboxes[1]
#     [0, 2, 2, 4]
#     >>> bboxes[2]
#     [1, 0, 3, 1]
#     >>> bboxes[3]
#     [3, 2, 4, 4]
#
#
#     :param labels: The output of cv2.connectedComponents().
#
#     :returns: A dict indexed by labels. The values are quadruplets
#         (xmin, ymin, xmax, ymax) so that the component with the given label
#         lies exactly within labels[xmin:xmax, ymin:ymax].
#     """
#     bboxes = {}
#     for x, row in enumerate(labels):
#         for y, l in enumerate(row):
#             if l not in bboxes:
#                 bboxes[l] = [x, y, x+1, y+1]
#             else:
#                 box = bboxes[l]
#                 if x < box[0]:
#                     box[0] = x
#                 elif x + 1 > box[2]:
#                     box[2] = x + 1
#                 if y < box[1]:
#                     box[1] = y
#                 elif y + 1 > box[3]:
#                     box[3] = y + 1
#     return bboxes
#
#
# def compute_connected_components(image):
#     labels = label(image, background=0)
#     cc = int(labels.max())
#     bboxes = connected_components2bboxes(labels)
#     return cc, labels, bboxes
#

##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--staff_imfile', action='store',
                        help='The image file with the staff-only image.'
                             ' If not given, will use -n, -w and -r'
                             ' to load it from the CVC-MUSCIMA staff removal'
                             ' ground truth.')

    parser.add_argument('-n', '--number', action='store', type=int,
                        help='Number of the CVC-MUSCIMA page (1 - 20)')
    parser.add_argument('-w', '--writer', action='store', type=int,
                        help='Writer of the CVC-MUSCIMA page (1 - 50)')

    parser.add_argument('-r', '--root', action='store',
                        default=os.getenv('CVC_MUSCIMA_ROOT', None),
                        help='Path to CVC-MUSCIMA dataset root. By default, will attempt'
                             ' to read the CVC_MUSCIMA_ROOT env var. If that does not'
                             ' work, the script will fail.')

    parser.add_argument('-a', '--annot', action='store', # required=True,
                        help='The annotation file for which the staffline and staff'
                             ' CropObjects should be added. If not supplied, default'
                             ' doc/collection names will be used and cropobjects will'
                             ' be numberd from 0 in the output.')
    parser.add_argument('-e', '--export', action='store',
                        help='A filename to which the output CropObjectList'
                             ' should be saved. If not given, will print to'
                             ' stdout.')
    parser.add_argument('--stafflines_only', action='store_true',
                        help='If set, will only output stafflines, not other symbols.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    ########################################################
    # Load gt image.
    logging.info('Loading staffline image.')
    #  - Initialize Dataset. This checks for the root.

    if args.staff_imfile is None:
        cvc_dataset = CVC_MUSCIMA(root=args.root)
        args.staff_imfile = cvc_dataset.imfile(page=args.number,
                                               writer=args.writer,
                                               distortion='ideal',
                                               mode='staff_only')

    # - Load the image.
    gt = (imread(args.staff_imfile, as_grey=True) * 255).astype('uint8')

    # - Cast as binary mask.
    gt[gt > 0] = 1

    ########################################################
    # Locate stafflines in gt image.
    logging.info('Getting staffline connected components.')

    #  - Get connected components in gt image.
    cc, labels, bboxes = compute_connected_components(gt)

    #  - Use vertical dimension of CCs to determine which ones belong together
    #    to form stafflines. (Criterion: row overlap.)
    n_rows, n_cols = gt.shape
    intervals = [[] for _ in range(n_rows)] # For each row: which CCs have pxs on that row?
    for label, (t, l, b, r) in list(bboxes.items()):
        if label == 0:
            continue
        # Ignore very short staffline segments that can easily be artifacts
        # and should not affect the vertical range of the staffline anyway.
        if (r - l) < 8:
            continue
        for row in range(t, b):
            intervals[row].append(label)

    logging.info('Grouping staffline connected components into stafflines.')
    staffline_components = []   # For each staffline, we collect the CCs that it is made of
    _in_staffline = False
    _current_staffline_components = []
    for r_labels in intervals:
        if not _in_staffline:
            # Last row did not contain staffline components.
            if len(r_labels) == 0:
                # No staffline component on current row
                continue
            else:
                _in_staffline = True
                _current_staffline_components += r_labels
        else:
            # Last row contained staffline components.
            if len(r_labels) == 0:
                # Current staffline has no more rows.
                staffline_components.append(set(_current_staffline_components))
                _current_staffline_components = []
                _in_staffline = False
                continue
            else:
                # Current row contains staffline components: the current
                # staffline continues.
                _current_staffline_components += r_labels

    logging.info('No. of stafflines, with component groups: {0}'
                 ''.format(len(staffline_components)))

    # Now: merge the staffline components into one bbox/mask.
    logging.info('Merging staffline components into staffline bboxes and masks.')
    staffline_bboxes = []
    staffline_masks = []
    for sc in sorted(staffline_components,
                     key=lambda c: min([bboxes[cc][0]
                                        for cc in c])):  # Sorted top-down
        st, sl, sb, sr = n_rows, n_cols, 0, 0
        for component in sc:
            t, l, b, r = bboxes[component]
            st, sl, sb, sr = min(t, st), min(l, sl), max(b, sb), max(r, sr)
        _sm = gt[st:sb, sl:sr]
        staffline_bboxes.append((st, sl, sb, sr))
        staffline_masks.append(_sm)

    # Check if n. of stafflines is divisible by 5
    n_stafflines = len(staffline_bboxes)
    logging.info('\tTotal stafflines: {0}'.format(n_stafflines))
    if n_stafflines % 5 != 0:
        import matplotlib.pyplot as plt
        stafllines_mask_image = numpy.zeros(gt.shape)
        for i, (_sb, _sm) in enumerate(zip(staffline_bboxes, staffline_masks)):
            t, l, b, r = _sb
            stafllines_mask_image[t:b, l:r] = min(255, (i * 333) % 255 + 40)
        plt.imshow(stafllines_mask_image, cmap='jet', interpolation='nearest')
        plt.show()
        raise ValueError('No. of stafflines is not divisible by 5!')

    logging.info('Creating staff bboxes and masks.')

    #  - Go top-down and group the stafflines by five to get staves.
    #    (The staffline bboxes are already sorted top-down.)
    staff_bboxes = []
    staff_masks = []

    for i in range(n_stafflines // 5):
        _sbb = staffline_bboxes[5*i:5*(i+1)]
        _st = min([bb[0] for bb in _sbb])
        _sl = min([bb[1] for bb in _sbb])
        _sb = max([bb[2] for bb in _sbb])
        _sr = max([bb[3] for bb in _sbb])
        staff_bboxes.append((_st, _sl, _sb, _sr))
        staff_masks.append(gt[_st:_sb, _sl:_sr])

    logging.info('Total staffs: {0}'.format(len(staff_bboxes)))

    ##################################################################
    # (Optionally fill in missing pixels, based on full image.)
    logging.info('SKIP: fill in missing pixels based on full image.')
    #  - Load full image
    #  - Find gap regions
    #  - Obtain gap region masks from full image
    #  - Add gap region mask to staffline mask.

    # Create the CropObjects for stafflines and staffs:
    #  - Load corresponding annotation, to which the stafflines and
    #    staves should be added. (This is needed to correctly set docname
    #    and objids.)
    if not args.annot:
        cropobjects = []
        next_objid = 0
        dataset_namespace = 'FCNOMR'
        docname = os.path.splitext(os.path.basename(args.staff_imfile))[0]
    else:
        if not os.path.isfile(args.annot):
            raise ValueError('Annotation file {0} does not exist!'.format(args.annot))

        logging.info('Creating cropobjects...')
        cropobjects = parse_cropobject_list(args.annot)
        logging.info('Non-staffline cropobjects: {0}'.format(len(cropobjects)))
        next_objid = max([c.objid for c in cropobjects]) + 1
        dataset_namespace = cropobjects[0].dataset
        docname = cropobjects[0].doc

    #  - Create the staffline CropObjects
    staffline_cropobjects = []
    for sl_bb, sl_m in zip(staffline_bboxes, staffline_masks):
        uid = Node.build_unique_id(dataset_namespace, docname, next_objid)
        t, l, b, r = sl_bb
        c = Node(node_id=next_objid,
                 class_name=STAFFLINE_CLSNAME,
                 top=t, left=l, height=b - t, width=r - l,
                 mask=sl_m,
                 unique_id=uid)
        staffline_cropobjects.append(c)
        next_objid += 1

    if not args.stafflines_only:

        #  - Create the staff CropObjects
        staff_cropobjects = []
        for s_bb, s_m in zip(staff_bboxes, staff_masks):
            uid = Node.build_unique_id(dataset_namespace, docname, next_objid)
            t, l, b, r = s_bb
            c = Node(node_id=next_objid,
                     class_name=STAFF_CLSNAME,
                     top=t, left=l, height=b - t, width=r - l,
                     mask=s_m,
                     unique_id=uid)
            staff_cropobjects.append(c)
            next_objid += 1

        #  - Add the inlinks/outlinks
        for i, sc in enumerate(staff_cropobjects):
            sl_from = 5 * i
            sl_to = 5 * (i + 1)
            for sl in staffline_cropobjects[sl_from:sl_to]:
                sl.inlinks.append(sc.node_id)
                sc.outlinks.append(sl.node_id)

        # Add the staffspaces.
        staffspace_cropobjects = []
        for i, staff in enumerate(staff_cropobjects):
            current_stafflines = [sc for sc in staffline_cropobjects if sc.node_id in staff.outlinks]
            sorted_stafflines = sorted(current_stafflines, key=lambda x: x.top)

            current_staffspace_cropobjects = []

            # Percussion single-line staves do not have staffspaces.
            if len(sorted_stafflines) == 1:
                continue

            # Internal staffspace
            for s1, s2 in zip(sorted_stafflines[:-1], sorted_stafflines[1:]):
                # s1 is the UPPER staffline, s2 is the LOWER staffline
                # Left and right limits: to simplify things, we take the column
                # *intersection* of (s1, s2). This gives the invariant that
                # the staffspace is limited from top and bottom in each of its columns.
                l = max(s1.left, s2.left)
                r = min(s1.right, s2.right)

                # Shift s1, s2 to the right by this much to have the cols. align
                # All of these are non-negative.
                dl1, dl2 = l - s1.left, l - s2.left
                dr1, dr2 = s1.right - r, s2.right - r

                # The stafflines are not necessarily straight,
                # so top is given for the *topmost bottom edge* of the top staffline + 1

                # First create mask
                canvas = numpy.zeros((s2.bottom - s1.top, r - l), dtype='uint8')

                # Paste masks into canvas.
                # This assumes that the top of the bottom staffline is below
                # the top of the top staffline... and that the bottom
                # of the top staffline is above the bottom of the bottom
                # staffline. This may not hold in very weird situations,
                # but it's good for now.
                logging.debug(s1.bounding_box, s1.mask.shape)
                logging.debug(s2.bounding_box, s2.mask.shape)
                logging.debug(canvas.shape)
                logging.debug('l={0}, dl1={1}, dl2={2}, r={3}, dr1={4}, dr2={5}'
                              ''.format(l, dl1, dl2, r, dr1, dr2))
                #canvas[:s1.height, :] += s1.mask[:, dl1:s1.width-dr1]
                #canvas[-s2.height:, :] += s2.mask[:, dl2:s2.width-dr2]

                # We have to deal with staffline interruptions.
                # One way to do this
                # is watershed fill: put markers along the bottom and top
                # edge, use mask * 10000 as elevation

                s1_above, s1_below = staffline_surroundings_mask(s1)
                s2_above, s2_below = staffline_surroundings_mask(s2)

                # Get bounding boxes of the individual stafflines' masks
                # that intersect with the staffspace bounding box, in terms
                # of the staffline bounding box.
                s1_t, s1_l, s1_b, s1_r = 0, dl1, \
                                         s1.height, s1.width - dr1
                s1_h, s1_w = s1_b - s1_t, s1_r - s1_l
                s2_t, s2_l, s2_b, s2_r = canvas.shape[0] - s2.height, dl2, \
                                         canvas.shape[0], s2.width - dr2
                s2_h, s2_w = s2_b - s2_t, s2_r - s2_l

                logging.debug(s1_t, s1_l, s1_b, s1_r, (s1_h, s1_w))

                # We now take the intersection of s1_below and s2_above.
                # If there is empty space in the middle, we fill it in.
                staffspace_mask = numpy.ones(canvas.shape)
                staffspace_mask[s1_t:s1_b, :] -= (1 - s1_below[:, dl1:s1.width-dr1])
                staffspace_mask[s2_t:s2_b, :] -= (1 - s2_above[:, dl2:s2.width-dr2])

                ss_top = s1.top
                ss_bottom = s2.bottom
                ss_left = l
                ss_right = r

                uid = Node.build_unique_id(dataset_namespace, docname, next_objid)

                staffspace = Node(next_objid, STAFFSPACE_CLSNAME,
                                  top=ss_top, left=ss_left,
                                  height=ss_bottom - ss_top,
                                  width=ss_right - ss_left,
                                  mask=staffspace_mask,
                                  unique_id=uid)

                staffspace.inlinks.append(staff.node_id)
                staff.outlinks.append(staffspace.node_id)

                current_staffspace_cropobjects.append(staffspace)

                next_objid += 1

            # Add top and bottom staffspace.
            # These outer staffspaces will have the width
            # of their bottom neighbor, and height derived
            # from its mask columns.
            # This is quite approximate, but it should do.

            # Upper staffspace
            tsl = sorted_stafflines[0]
            tsl_heights = tsl.mask.sum(axis=0)
            tss = current_staffspace_cropobjects[0]
            tss_heights = tss.mask.sum(axis=0)

            uss_top = max(0, tss.top - max(tss_heights))
            uss_left = tss.left
            uss_width = tss.width
            # We use 1.5, so that large noteheads
            # do not "hang out" of the staffspace.
            uss_height = int(tss.height / 1.2)
            # Shift because of height downscaling:
            uss_top += tss.height - uss_height
            uss_mask = tss.mask[:uss_height, :] * 1

            uid = Node.build_unique_id(dataset_namespace, docname, next_objid)
            staffspace = Node(next_objid, STAFFSPACE_CLSNAME,
                              top=uss_top, left=uss_left,
                              height=uss_height,
                              width=uss_width,
                              mask=uss_mask,
                              unique_id=uid)
            current_staffspace_cropobjects.append(staffspace)
            staff.outlinks.append(staffspace.node_id)
            staffspace.inlinks.append(staff.node_id)
            next_objid += 1

            # Lower staffspace
            bss = current_staffspace_cropobjects[-1]
            bss_heights = bss.mask.sum(axis=0)
            bsl = sorted_stafflines[-1]
            bsl_heights = bsl.mask.sum(axis=0)

            lss_top = bss.bottom # + max(bsl_heights)
            lss_left = bss.left
            lss_width = bss.width
            lss_height = int(bss.height / 1.2)
            lss_mask = bss.mask[:lss_height, :] * 1

            uid = Node.build_unique_id(dataset_namespace, docname, next_objid)
            staffspace = Node(next_objid, STAFFSPACE_CLSNAME,
                              top=lss_top, left=lss_left,
                              height=lss_height,
                              width=lss_width,
                              mask=lss_mask,
                              unique_id=uid)
            current_staffspace_cropobjects.append(staffspace)
            staff.outlinks.append(staffspace.node_id)
            staffspace.inlinks.append(staff.node_id)
            next_objid += 1

            # ################ End of dealing with upper/lower staffspace ######

            # Add to current list
            staffspace_cropobjects += current_staffspace_cropobjects

        # - Join the lists together
        cropobjects_with_staffs = cropobjects \
                                  + staffline_cropobjects \
                                  + staffspace_cropobjects \
                                  + staff_cropobjects

    else:
        cropobjects_with_staffs = cropobjects + staffline_cropobjects

    logging.info('Exporting the new cropobject list: {0} objects'
                    ''.format(len(cropobjects_with_staffs)))
    # - Export the combined list.
    cropobject_string = export_cropobject_list(cropobjects_with_staffs)
    if args.export is not None:
        with open(args.export, 'w') as hdl:
            hdl.write(cropobject_string)
    else:
        print(cropobject_string)

    _end_time = time.clock()
    logging.info('add_staffline_symbols.py done in {0:.3f} s'
                    ''.format(_end_time - _start_time))


def staffline_surroundings_mask(staffline_cropobject):
    """Find the parts of the staffline's bounding box which lie
    above or below the actual staffline.

    These areas will be very small for straight stafflines,
    but might be considerable when staffline curvature grows.
    """
    # We segment both masks into "above staffline" and "below staffline"
    # areas.
    elevation = staffline_cropobject.mask * 255
    # Blur, to plug small holes somewhat:
    elevation = gaussian(elevation, sigma=1.0)
    # Prepare the segmentation markers: 1 is ABOVE, 2 is BELOW
    markers = numpy.zeros(staffline_cropobject.mask.shape)
    markers[0, :] = 1
    markers[-1, :] = 2
    markers[staffline_cropobject.mask != 0] = 0
    seg = watershed(elevation, markers)

    bmask = numpy.ones(seg.shape)
    bmask[seg != 2] = 0
    tmask = numpy.ones(seg.shape)
    tmask[seg != 1] = 0

    return bmask, tmask


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
