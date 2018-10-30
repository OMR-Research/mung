"""This module implements functions for manipulating staffline symbols.
Mostly it is support for going from detected staffline fragments
to full staff objects and relationships; this machinery is called
e.g. by pressing "shift+s" in MUSCIMarker."""
from __future__ import print_function, unicode_literals, division

from builtins import zip
from builtins import range
import collections
import logging
import pprint

import numpy
from skimage.filters import gaussian
from skimage.morphology import watershed

from mung.node import CropObject, cropobjects_on_canvas, link_cropobjects
from mung.graph import NotationGraph, find_noteheads_on_staff_linked_to_ledger_line
from mung.inference_engine_constants import InferenceEngineConstants as _CONST
from mung.utils import compute_connected_components


__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


def __has_parent_staff(c, cropobjects):
    _cdict = {c.objid: c for c in cropobjects}
    staff_inlinks = [_cdict[i] for i in c.inlinks
                     if _cdict[i].clsname == _CONST.STAFF_CLSNAME]
    return len(staff_inlinks) > 0


def __has_child_staffspace(staff, cropobjects):
    _cdict = {c.objid: c for c in cropobjects}
    staffline_outlinks = [_cdict[i] for i in staff.outlinks
                          if _cdict[i].clsname == _CONST.STAFFSPACE_CLSNAME]
    return len(staffline_outlinks) > 0


def __has_neighbor_staffspace(staffline, cropobjects):
    _cdict = {c.objid: c for c in cropobjects}
    # Find parent staff
    if not __has_parent_staff(staffline, cropobjects):
        return False
    parent_staffs = [_cdict[i] for i in staffline.inlinks
                     if _cdict[i].clsname == _CONST.STAFF_CLSNAME]
    if len(parent_staffs) > 1:
        raise ValueError('More than one parent staff for staffline {0}!'
                         ''.format(staffline.uid))
    staff = parent_staffs[0]
    return __has_child_staffspace(staff, cropobjects)


##############################################################################


def merge_staffline_segments(cropobjects, margin=10):
    """Given a list of CropObjects that contain some staffline
    objects, generates a new list where the stafflines
    are merged based on their horizontal projections.
    Basic step for going from the staffline detection masks to
    the actual staffline objects.

    Assumes that stafflines are straight: their bounding boxes
    do not touch or overlap.

    :param cropobjects:
    :param margin:

    :return: A modified CropObject list: the original staffline-class
        symbols are replaced by the merged ones. If the original stafflines
        had any inlinks, they are preserved (mapped to the new staffline).
    """
    already_processed_stafflines = [c for c in cropobjects
                                    if (c.clsname == _CONST.STAFFLINE_CLSNAME) and
                                        __has_parent_staff(c, cropobjects)]
    # margin is used to avoid the stafflines touching the edges,
    # which could perhaps break some assumptions down the line.
    old_staffline_cropobjects = [c for c in cropobjects
                                 if (c.clsname == _CONST.STAFFLINE_CLSNAME) and
                                 not __has_parent_staff(c, cropobjects)]
    if len(old_staffline_cropobjects) == 0:
        logging.info('merge_staffline_segments: nothing new to do!')
        return cropobjects

    canvas, (_t, _l) = cropobjects_on_canvas(old_staffline_cropobjects)

    _staffline_bboxes, staffline_masks = staffline_bboxes_and_masks_from_horizontal_merge(canvas)
    # Bounding boxes need to be adjusted back with respect to the original image!
    staffline_bboxes = [(t + _t, l + _l, b + _t, r + _l) for t, l, b, r in _staffline_bboxes]

    # Create the CropObjects.
    next_objid = max([c.objid for c in cropobjects]) + 1
    dataset_namespace = cropobjects[0].dataset
    docname = cropobjects[0].doc

    #  - Create the staffline CropObjects
    staffline_cropobjects = []
    for sl_bb, sl_m in zip(staffline_bboxes, staffline_masks):
        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        t, l, b, r = sl_bb
        c = CropObject(objid=next_objid,
                       clsname=_CONST.STAFFLINE_CLSNAME,
                       top=t, left=l, height=b - t, width=r - l,
                       mask=sl_m,
                       uid=uid)
        staffline_cropobjects.append(c)
        next_objid += 1

    non_staffline_cropobjects = [c for c in cropobjects
                                 if c.clsname != _CONST.STAFFLINE_CLSNAME]
    old_staffline_objids = set([c.objid for c in old_staffline_cropobjects])
    old2new_staffline_objid_map = {}
    for os in old_staffline_cropobjects:
        for ns in staffline_cropobjects:
            if os.overlaps(ns):
                old2new_staffline_objid_map[os.objid] = ns

    logging.info('Re-linking from the old staffline objects to new ones.')
    for c in non_staffline_cropobjects:
        new_outlinks = []
        for o in c.outlinks:
            if o in old_staffline_objids:
                new_staffline = old2new_staffline_objid_map[o]
                new_outlinks.append(new_staffline.objid)
                new_staffline.inlinks.append(c.objid)
            else:
                new_outlinks.append(o)

    output = non_staffline_cropobjects + staffline_cropobjects + already_processed_stafflines
    return output


def staffline_bboxes_and_masks_from_horizontal_merge(mask):
    """Returns a list of staff_line bboxes and masks
     computed from the input mask, with
    each set of connected components in the mask that has at least
    one pixel in a neighboring or overlapping row is assigned to
    the same label. Intended for finding staffline masks from individual
    components of the stafflines (for this purpose, you have to assume
    that the stafflines are straight)."""
    logging.info('Getting staffline connected components.')

    cc, labels, bboxes = compute_connected_components(mask)

    logging.info('Getting staffline component vertical projections')
    #  - Use vertical dimension of CCs to determine which ones belong together
    #    to form stafflines. (Criterion: row overlap.)
    n_rows, n_cols = mask.shape
    # For each row of the image: which CCs have pxs on that row?
    intervals = [[] for _ in range(n_rows)]
    for label, (t, l, b, r) in list(bboxes.items()):
        if label == 0:
            continue
        # Ignore very short staffline segments that can easily be artifacts
        # and should not affect the vertical range of the staffline anyway.
        # The "8" is a magic number, no good reason for specifically 8.
        # (It should be some proportion of the image width, like 0.01.)
        if (r - l) < 8:
            continue
        for row in range(t, b):
            intervals[row].append(label)

    logging.warning('Grouping staffline connected components into stafflines.')
    # For each staffline, we collect the CCs that it is made of. We assume stafflines
    # are separated from each other by at least one empty row.
    staffline_components = []
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
        # Here, again, we assume that no two "real" stafflines have overlapping
        # bounding boxes.
        _sm = mask[st:sb, sl:sr]
        staffline_bboxes.append((st, sl, sb, sr))
        staffline_masks.append(_sm)

    # Check if n. of stafflines is divisible by 5
    n_stafflines = len(staffline_bboxes)
    logging.warning('\tTotal stafflines: {0}'.format(n_stafflines))
    if n_stafflines % 5 != 0:
        # try:
        #     import matplotlib.pyplot as plt
        #     stafllines_mask_image = numpy.zeros(mask.shape)
        #     for i, (_sb, _sm) in enumerate(zip(staffline_bboxes, staffline_masks)):
        #         t, l, b, r = _sb
        #         stafllines_mask_image[t:b, l:r] = min(255, (i * 333) % 255 + 40)
        #     plt.imshow(stafllines_mask_image, cmap='jet', interpolation='nearest')
        #     plt.show()
        # except ImportError:
        #     pass
        raise ValueError('No. of stafflines is not divisible by 5!')

    return staffline_bboxes, staffline_masks


def staff_bboxes_and_masks_from_staffline_bboxes_and_image(staffline_bboxes, mask):
    logging.warning('Creating staff bboxes and masks.')

    #  - Go top-down and group the stafflines by five to get staves.
    #    (The staffline bboxes are already sorted top-down.)
    staff_bboxes = []
    staff_masks = []

    n_stafflines = len(staffline_bboxes)
    for i in range(n_stafflines // 5):
        _sbb = staffline_bboxes[5*i:5*(i+1)]
        _st = min([bb[0] for bb in _sbb])
        _sl = min([bb[1] for bb in _sbb])
        _sb = max([bb[2] for bb in _sbb])
        _sr = max([bb[3] for bb in _sbb])
        staff_bboxes.append((_st, _sl, _sb, _sr))
        staff_masks.append(mask[_st:_sb, _sl:_sr])

    logging.warning('Total staffs: {0}'.format(len(staff_bboxes)))

    return staff_bboxes, staff_masks


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


##############################################################################


def build_staff_cropobjects(cropobjects):
    """Derives staff objects from staffline objcets.

    Assumes each staff has 5 stafflines.

    Assumes the stafflines have already been merged."""
    stafflines = [c for c in cropobjects
                  if c.clsname == _CONST.STAFFLINE_CLSNAME and
                    not __has_parent_staff(c, cropobjects)]
    if len(stafflines) == 0:
        return []

    staffline_bboxes = [c.bounding_box for c in stafflines]
    canvas, (_t, _l) = cropobjects_on_canvas(stafflines)

    logging.warning('Creating staff bboxes and masks.')

    #  - Go top-down and group the stafflines by five to get staves.
    #    (The staffline bboxes are already sorted top-down.)
    staff_bboxes = []
    staff_masks = []

    n_stafflines = len(stafflines)
    for i in range(n_stafflines // 5):
        _sbb = staffline_bboxes[5*i:5*(i+1)]
        _st = min([bb[0] for bb in _sbb])
        _sl = min([bb[1] for bb in _sbb])
        _sb = max([bb[2] for bb in _sbb])
        _sr = max([bb[3] for bb in _sbb])
        staff_bboxes.append((_st, _sl, _sb, _sr))
        staff_masks.append(canvas[_st-_t:_sb-_t, _sl-_l:_sr-_l])

    logging.info('Creating staff CropObjects')
    next_objid = max([c.objid for c in cropobjects]) + 1
    dataset_namespace = cropobjects[0].dataset
    docname = cropobjects[0].doc

    staff_cropobjects = []
    for s_bb, s_m in zip(staff_bboxes, staff_masks):
        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        t, l, b, r = s_bb
        c = CropObject(objid=next_objid,
                       clsname=_CONST.STAFF_CLSNAME,
                       top=t, left=l, height=b - t, width=r - l,
                       mask=s_m,
                       uid=uid)
        staff_cropobjects.append(c)
        next_objid += 1

    for i, sc in enumerate(staff_cropobjects):
        sl_from = 5 * i
        sl_to = 5 * (i + 1)
        for sl in stafflines[sl_from:sl_to]:
            sl.inlinks.append(sc.objid)
            sc.outlinks.append(sl.objid)

    return staff_cropobjects


def build_staffspace_cropobjects(cropobjects):
    """Creates the staffspace objects based on stafflines
    and staffs. There is a staffspace between each two stafflines,
    one on the top side of each staff, and one on the bottom
    side for each staff (corresponding e.g. to positions of g5 and d4
    with the standard G-clef).

    Note that staffspaces do not assume anything about the number
    of stafflines per staff. However, single-staffline staffs will
    not have the outer staffspaces generated (there is nothing to derive
    their size from), for now.

    :param cropobjects: A list of CropObjects that must contain
        all the relevant stafflines and staffs.

    :return: A list of staffspace CropObjects.
    """
    next_objid = max([c.objid for c in cropobjects]) + 1
    dataset_namespace = cropobjects[0].dataset
    docname = cropobjects[0].doc

    staff_cropobjects = [c for c in cropobjects
                         if c.clsname == _CONST.STAFF_CLSNAME
                         and not __has_child_staffspace(c, cropobjects)]
    staffline_cropobjects = [c for c in cropobjects
                             if c.clsname == _CONST.STAFFLINE_CLSNAME
                             and not __has_neighbor_staffspace(c, cropobjects)]

    staffspace_cropobjects = []

    for i, staff in enumerate(staff_cropobjects):
        current_stafflines = [sc for sc in staffline_cropobjects
                              if sc.objid in staff.outlinks]
        sorted_stafflines = sorted(current_stafflines, key=lambda x: x.top)

        current_staffspace_cropobjects = []

        # Percussion single-line staves do not have staffspaces.
        if len(sorted_stafflines) == 1:
            continue

        #################
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

            uid = CropObject.build_uid(dataset_namespace, docname, next_objid)

            staffspace = CropObject(next_objid, _CONST.STAFFSPACE_CLSNAME,
                                    top=ss_top, left=ss_left,
                                    height=ss_bottom - ss_top,
                                    width=ss_right - ss_left,
                                    mask=staffspace_mask,
                                    uid=uid)

            staffspace.inlinks.append(staff.objid)
            staff.outlinks.append(staffspace.objid)

            current_staffspace_cropobjects.append(staffspace)

            next_objid += 1

        ##########
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

        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        staffspace = CropObject(next_objid, _CONST.STAFFSPACE_CLSNAME,
                                top=uss_top, left=uss_left,
                                height=uss_height,
                                width=uss_width,
                                mask=uss_mask,
                                uid=uid)
        current_staffspace_cropobjects.append(staffspace)
        staff.outlinks.append(staffspace.objid)
        staffspace.inlinks.append(staff.objid)
        next_objid += 1

        # Lower staffspace
        bss = current_staffspace_cropobjects[-2]
        bss_heights = bss.mask.sum(axis=0)
        bsl = sorted_stafflines[-1]
        bsl_heights = bsl.mask.sum(axis=0)

        lss_top = bss.bottom # + max(bsl_heights)
        lss_left = bss.left
        lss_width = bss.width
        lss_height = int(bss.height / 1.2)
        lss_mask = bss.mask[:lss_height, :] * 1

        uid = CropObject.build_uid(dataset_namespace, docname, next_objid)
        staffspace = CropObject(next_objid, _CONST.STAFFSPACE_CLSNAME,
                                top=lss_top, left=lss_left,
                                height=lss_height,
                                width=lss_width,
                                mask=lss_mask,
                                uid=uid)
        current_staffspace_cropobjects.append(staffspace)
        staff.outlinks.append(staffspace.objid)
        staffspace.inlinks.append(staff.objid)
        next_objid += 1

        staffspace_cropobjects += current_staffspace_cropobjects

    return staffspace_cropobjects


##############################################################################


def add_staff_relationships(cropobjects,
                            notehead_staffspace_threshold=0.2,
                            reprocess_noteheads_inside_staff_with_lls=True):
    """Adds the relationships from various symbols to staff objects:
    stafflines, staffspaces, and staffs.

    :param cropobjects: The list of cropobjects in the document. Must
        include the staff objects.

    :param notehead_staffspace_threshold: A notehead is considered to be
        on a staffline if it intersects a staffline, and the ratio between
        how far above the staffline and how far below the staffline it reaches
        is at least this (default: 0.2). The ratio is computed both ways:
        d_top / d_bottom and d_bottom / d_top, and the minimum is taken,
        so the default in effect restricts d_top / d_bottom between 0.2 and 0.8:
        in other words, the imbalance of the notehead's bounding box around
        the staffline should be less than 1:4.

    :param reprocess_noteheads_inside_staff_with_lls: If set to True, will check against noteheads
        that are connected to ledger lines, but intersect a staffline. If found,
        will remove their edges before further processing, so that the noteheads
        will seem properly unprocessed.

        Note that this handling is a bit ad-hoc for now. However, we currently
        do not have a better place to fit this in, since the Best Practices
        currently call for first applying the syntactic parser and then
        adding staff relationships.

    :return: The list of cropobjects corresponding to the new graph.
    """
    graph = NotationGraph(cropobjects)
    _cropobjects_dict = {c.objid: c for c in cropobjects}

    ON_STAFFLINE_RATIO_TRHESHOLD = notehead_staffspace_threshold

    ##########################################################################
    if reprocess_noteheads_inside_staff_with_lls:
        ll_noteheads_on_staff = find_noteheads_on_staff_linked_to_ledger_line(cropobjects)
        logging.info('Reprocessing noteheads that are inside a staff, but have links'
                     ' to ledger lines. Found {0} such noteheads.'
                     ''.format(len(ll_noteheads_on_staff)))
        for n in ll_noteheads_on_staff:
            # Remove all links to ledger lines.
            lls = graph.children(n, classes=['ledger_line'])
            for ll in lls:
                graph.remove_edge(n.objid, ll.objid)

    ##########################################################################
    logging.info('Find the staff-related symbols')
    staffs = [c for c in cropobjects if c.clsname == _CONST.STAFF_CLSNAME]

    staff_related_symbols = collections.defaultdict(list)
    notehead_symbols = collections.defaultdict(list)
    rest_symbols = collections.defaultdict(list)
    for c in cropobjects:
        if c.clsname in _CONST.STAFF_RELATED_CLSNAMES:
            # Check if it already has a staff link
            if not graph.has_child(c, [_CONST.STAFF_CLSNAME]):
                staff_related_symbols[c.clsname].append(c)
        if c.clsname in _CONST.NOTEHEAD_CLSNAMES:
            if not graph.has_child(c, [_CONST.STAFF_CLSNAME]):
                notehead_symbols[c.clsname].append(c)
        if c.clsname in _CONST.REST_CLSNAMES:
            if not graph.has_child(c, [_CONST.STAFF_CLSNAME]):
                rest_symbols[c.clsname].append(c)

    ##########################################################################
    logging.info('Adding staff relationships')
    #  - Which direction do the relationships lead in?
    #    Need to define this.
    #
    # Staff -> symbol?
    # Symbol -> staff?
    # It does not really matter, but it's more intuitive to attach symbols
    # onto a pre-existing staff. So, symbol -> staff.
    for clsname, cs in list(staff_related_symbols.items()):
        for c in cs:
            # Find the related staff. Relatedness is measured by row overlap.
            # That means we have to modify the staff bounding box to lead
            # from the leftmost to the rightmost column. This holds
            # especially for the staff_grouping symbols.
            for s in staffs:
                st, sl, sb, sr = s.bounding_box
                sl = 0
                sr = max(sr, c.right)
                if c.overlaps((st, sl, sb, sr)):
                    link_cropobjects(c, s, check_docname=False)

    ##########################################################################
    logging.info('Adding rest --> staff relationships.')
    for clsname, cs in list(rest_symbols.items()):
        for c in cs:
            closest_staff = min([s for s in staffs],
                                key=lambda x: ((x.bottom + x.top) / 2. - (c.bottom + c.top) / 2.) ** 2)
            link_cropobjects(c, closest_staff, check_docname=False)

    ##########################################################################
    logging.info('Adding notehead relationships.')

    # NOTE:
    # This part should NOT rely on staffspace masks in any way!
    # They are highly unreliable.

    # Sort the staff objects top-down. Assumes stafflines do not cross,
    # and that there are no crazy curves at the end that would make the lower
    # stafflines stick out over the ones above them...
    stafflines = [c for c in cropobjects if c.clsname == _CONST.STAFFLINE_CLSNAME]
    stafflines = sorted(stafflines, key=lambda c: c.top)
    staffspaces = [c for c in cropobjects if c.clsname == _CONST.STAFFSPACE_CLSNAME]
    staffspaces= sorted(staffspaces, key=lambda c: c.top)
    staves = [c for c in cropobjects if c.clsname == _CONST.STAFF_CLSNAME]
    staves = sorted(staves, key=lambda c: c.top)

    logging.info('Stafflines: {0}'.format(len(stafflines)))
    logging.info('Staffspaces: {0}'.format(len(staffspaces)))
    logging.info('Staves: {0}'.format(len(staves)))

    # Indexing data structures.
    #
    # We need to know:
    #  - per staffline and staffspace: its containing staff
    _staff_per_ss_sl = {}
    #  - per staffline and staffspace: its index (top to bottom) within the staff
    _ss_sl_idx_wrt_staff = {}
    # Reverse indexes:
    # If I know which staff (by objid) and which index of staffline/staffspace,
    # I want to retrieve the given staffline/staffspace CropObject:
    _staff_and_idx2ss = collections.defaultdict(dict)
    _staff_and_idx2sl = collections.defaultdict(dict)

    # Build the indexes
    for _staff in staves:
        # Keep the top-down ordering from above:
        _s_stafflines = [_staffline for _staffline in stafflines
                         if _staff.objid in _staffline.inlinks]
        _s_staffspaces = [_staffspace for _staffspace in staffspaces
                          if _staff.objid in _staffspace.inlinks]
        for i, _sl in enumerate(_s_stafflines):
            _staff_per_ss_sl[_sl.objid] = _staff
            _ss_sl_idx_wrt_staff[_sl.objid] = i
            _staff_and_idx2sl[_staff.objid][i] = _sl
            logging.debug('Staff {0}: stafflines {1}'.format(_staff.objid,
                                                             _staff_and_idx2sl[_staff.objid]))
        for i, _ss in enumerate(_s_staffspaces):
            _staff_per_ss_sl[_ss.objid] = _staff
            _ss_sl_idx_wrt_staff[_ss.objid] = i
            _staff_and_idx2ss[_staff.objid][i] = _ss

    # pprint.pprint(_ss_sl_idx_wrt_staff)
    logging.debug(pprint.pformat(dict(_staff_and_idx2ss)))
    # pprint.pprint(dict(_staff_and_idx2sl))

    # # Get bounding box of all participating symbols
    # notehead_staff_bbox_coords = [c.bounding_box
    #                               for c in list(itertools.chain(*notehead_symbols.values()))
    #                                        + stafflines
    #                                        + staffspaces
    #                                        + staves]
    # t, l, b, r = min([b[0] for b in notehead_staff_bbox_coords]), \
    #              min([b[1] for b in notehead_staff_bbox_coords]), \
    #              max([b[2] for b in notehead_staff_bbox_coords]), \
    #              max([b[3] for b in notehead_staff_bbox_coords])
    #
    # h, w = b - t, r - l
    # # Maybe later: ensure a 1-px empty border? (h+2, w+2) and the appropriate +1 when
    # # projecting cropobjects onto the canvas...
    # staffline_canvas = numpy.zeros((h, w), dtype='uint8')
    # for s in stafflines:
    #     dt, dl, db, dr = s.top - t, s.left - l, s.bottom - t, s.right - l
    #     staffline_canvas[dt:db, dl:dr] += s.mask

    for clsname, cs in list(notehead_symbols.items()):
        for c in cs:

            ct, cl, cb, cr = c.bounding_box

            ################
            # Add relationship to given staffline or staffspace.

            # If notehead has ledger lines, skip it for now.
            _has_ledger_line = False
            for o in c.outlinks:
                if _cropobjects_dict[o].clsname == 'ledger_line':
                    _has_ledger_line = True
                    break

            if _has_ledger_line:
                # Attach to the appropriate staff:
                # meaning, staff closest to the innermost ledger line.
                lls = [_cropobjects_dict[o] for o in c.outlinks
                       if _cropobjects_dict[o].clsname == 'ledger_line']
                # Furthest from notehead's top is innermost.
                # (If notehead is below staff and crosses a ll., one
                #  of these numbers will be negative. But that doesn't matter.)
                ll_max_dist = max(lls, key=lambda ll: ll.top - c.top)
                # Find closest staff to max-dist ledger ine
                staff_min_dist = min(staves,
                                     key=lambda ss: min((ll_max_dist.bottom - ss.top) ** 2,
                                                        (ll_max_dist.top - ss.bottom) ** 2))
                distance_of_closest_staff = (ll_max_dist.top + ll_max_dist.bottom) / 2 \
                                    - (staff_min_dist.top + staff_min_dist.bottom) / 2
                if numpy.abs(distance_of_closest_staff) > (50 + 0.5 * staff_min_dist.height):
                    logging.debug('Trying to join notehead with ledger line to staff,'
                                  ' but the distance is larger than 50. Notehead: {0},'
                                  ' ledger line: {1}, staff: {2}, distance: {3}'
                                  ''.format(c.uid, ll_max_dist.uid, staff_min_dist.uid,
                                            distance_of_closest_staff))
                else:
                    link_cropobjects(c, staff_min_dist, check_docname=False)
                continue

            # - Find the related staffline.
            # - Because of curved stafflines, this has to be done w.r.t.
            #   the horizontal position of the notehead.
            # - Also, because stafflines are NOT filled in (they do not have
            #   intersections annotated), it is necessary to use a wider
            #   window than just the notehead.
            # - We will assume that STAFFLINES DO NOT CROSS.
            #   (That is a reasonable assumption.)
            #
            # - For now, we only work with more or less straight stafflines.

            overlapped_stafflines = []
            overlapped_staffline_idxs = []
            for i, s in enumerate(stafflines):
                # This is the assumption of straight stafflines!
                if (ct <= s.top <= cb) or (ct <= s.bottom <= cb):
                    overlapped_stafflines.append(s)
                    overlapped_staffline_idxs.append(i)

            if c.objid < 10:
                logging.info('Notehead {0} ({1}): overlaps {2} stafflines'.format(c.uid,
                                                                                   c.bounding_box,
                                                                                   len(overlapped_stafflines),
                                                                                   ))

            if len(overlapped_stafflines) == 1:
                s = overlapped_stafflines[0]
                dtop = s.top - ct
                dbottom = cb - s.bottom

                logging.info('Notehead {0}, staffline {1}: ratio {2:.2f}'
                             ''.format(c.objid, s.objid, min(dtop, dbottom) / max(dtop, dbottom)))
                if min(dtop, dbottom) / max(dtop, dbottom) < ON_STAFFLINE_RATIO_TRHESHOLD:
                    # Staffspace?
                    #
                    # To get staffspace:
                    #  - Get orientation (below? above?)
                    _is_staffspace_above = False
                    if dtop > dbottom:
                        _is_staffspace_above = True

                    #  - Find staffspaces adjacent to the overlapped staffline.
                    # NOTE: this will fail with single-staffline staves, because
                    #       they do NOT have the surrounding staffspaces defined...
                    _staffline_idx_wrt_staff = _ss_sl_idx_wrt_staff[s.objid]
                    if _is_staffspace_above:
                        _staffspace_idx_wrt_staff = _staffline_idx_wrt_staff
                    else:
                        _staffspace_idx_wrt_staff = _staffline_idx_wrt_staff + 1

                    # Retrieve the given staffsapce
                    _staff = _staff_per_ss_sl[s.objid]
                    tgt_staffspace = _staff_and_idx2ss[_staff.objid][_staffspace_idx_wrt_staff]
                    # Link to staffspace
                    link_cropobjects(c, tgt_staffspace, check_docname=False)
                    # And link to staff
                    _c_staff = _staff_per_ss_sl[tgt_staffspace.objid]
                    link_cropobjects(c, _c_staff, check_docname=False)

                else:
                    # Staffline!
                    link_cropobjects(c, s, check_docname=False)
                    # And staff:
                    _c_staff = _staff_per_ss_sl[s.objid]
                    link_cropobjects(c, _c_staff, check_docname=False)
            elif len(overlapped_stafflines) == 0:
                # Staffspace!
                # Link to the staffspace with which the notehead has
                # greatest vertical overlap.
                #
                # Interesting corner case:
                # Sometimes noteheads "hang out" of the upper/lower
                # staffspace, so they are not entirely covered.
                overlapped_staffspaces = {}
                for _ss_i, s in enumerate(staffspaces):
                    if s.top <= c.top <= s.bottom:
                        overlapped_staffspaces[_ss_i] = min(s.bottom, c.bottom) - c.top
                    elif c.top <= s.top <= c.bottom:
                        overlapped_staffspaces[_ss_i] = s.bottom - max(c.top, s.top)

                if len(overlapped_staffspaces) == 0:
                    logging.warn('Notehead {0}: no overlapped staffline object, no ledger line!'
                                 ' Expecting it will be attached to ledger line and staff later on.'
                                 ''.format(c.uid))
                    continue

                _ss_i_max = max(list(overlapped_staffspaces.keys()),
                                key=lambda x: overlapped_staffspaces[x])
                max_overlap_staffspace = staffspaces[_ss_i_max]
                link_cropobjects(c, max_overlap_staffspace, check_docname=False)
                _c_staff = _staff_per_ss_sl[max_overlap_staffspace.objid]
                link_cropobjects(c, _c_staff, check_docname=False)

            elif len(overlapped_stafflines) == 2:
                # Staffspace between those two lines.
                s1 = overlapped_stafflines[0]
                s2 = overlapped_stafflines[1]

                _staff1 = _staff_per_ss_sl[s1.objid]
                _staff2 = _staff_per_ss_sl[s2.objid]
                if _staff1.objid != _staff2.objid:
                    raise ValueError('Really weird notehead overlapping two stafflines'
                                     ' from two different staves: {0}'.format(c.uid))

                _staffspace_idx = _ss_sl_idx_wrt_staff[s2.objid]
                s = _staff_and_idx2ss[_staff2.objid][_staffspace_idx]
                link_cropobjects(c, s, check_docname=False)
                # And link to staff:
                _c_staff = _staff_per_ss_sl[s.objid]
                link_cropobjects(c, _c_staff, check_docname=False)

            elif len(overlapped_stafflines) > 2:
                logging.warning('Really weird notehead overlapping more than 2 stafflines:'
                                ' {0} (permissive: linking to middle staffline)'.format(c.uid))
                # use the middle staffline -- this will be an error anyway,
                # but we want to export some MIDI more or less no matter what
                s_middle = overlapped_stafflines[len(overlapped_stafflines) // 2]
                link_cropobjects(c, s_middle, check_docname=False)
                # And staff:
                _c_staff = _staff_per_ss_sl[s_middle.objid]
                link_cropobjects(c, _c_staff, check_docname=False)

    ##########################################################################
    logging.info('Attaching clefs to stafflines [NOT IMPLEMENTED].')
    clefs = [c for c in cropobjects if c.clsname in ['g-clef', 'c-clef', 'f-clef']]

    ##########################################################################
    return cropobjects

