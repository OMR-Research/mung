"""This module implements functions for manipulating staffline symbols.
Mostly it is support for going from detected staffline fragments
to full staff objects and relationships; this machinery is called
e.g. by pressing "shift+s" in MUSCIMarker."""
import collections
import logging
import pprint

import numpy
from skimage.filters import gaussian
from skimage.morphology import watershed
from typing import List, Tuple

from mung.graph import NotationGraph, find_noteheads_on_staff_linked_to_leger_line
from mung.constants import InferenceEngineConstants
from mung.node import Node, draw_nodes_on_empty_canvas, link_nodes
from mung.utils import compute_connected_components

_CONST = InferenceEngineConstants()


def __has_parent_staff(node: Node, nodes: List[Node]) -> bool:
    id_to_node_mapping = {c.id: c for c in nodes}
    staff_inlinks = [id_to_node_mapping[i] for i in node.inlinks
                     if id_to_node_mapping[i].class_name == _CONST.STAFF_CLASS_NAME]
    return len(staff_inlinks) > 0


def __has_child_staffspace(staff: Node, nodes: List[Node]) -> bool:
    id_to_node_mapping = {c.id: c for c in nodes}
    staffline_outlinks = [id_to_node_mapping[i] for i in staff.outlinks
                          if id_to_node_mapping[i].class_name == _CONST.STAFFSPACE_CLASS_NAME]
    return len(staffline_outlinks) > 0


def __has_neighbor_staffspace(staffline: Node, nodes: List[Node]) -> bool:
    id_to_node_mapping = {c.id: c for c in nodes}
    # Find parent staff
    if not __has_parent_staff(staffline, nodes):
        return False
    parent_staffs = [id_to_node_mapping[i] for i in staffline.inlinks
                     if id_to_node_mapping[i].class_name == _CONST.STAFF_CLASS_NAME]
    if len(parent_staffs) > 1:
        raise ValueError('More than one parent staff for staffline {0}!'
                         ''.format(staffline.id))
    staff = parent_staffs[0]
    return __has_child_staffspace(staff, nodes)


def merge_staffline_segments(nodes: List[Node], margin: int = 10) -> List[Node]:
    """Given a list of Nodes that contain some staffline
    objects, generates a new list where the stafflines
    are merged based on their horizontal projections.
    Basic step for going from the staffline detection masks to
    the actual staffline objects.

    Assumes that stafflines are straight: their bounding boxes
    do not touch or overlap.

    :return: A modified Node list: the original staffline-class
        symbols are replaced by the merged ones. If the original stafflines
        had any inlinks, they are preserved (mapped to the new staffline).
    """
    already_processed_stafflines = [node for node in nodes
                                    if (node.class_name == _CONST.STAFFLINE_CLASS_NAME) and
                                    __has_parent_staff(node, nodes)]
    # margin is used to avoid the stafflines touching the edges,
    # which could perhaps break some assumptions down the line.
    old_staffline_nodes = [c for c in nodes
                           if (c.class_name == _CONST.STAFFLINE_CLASS_NAME) and
                           not __has_parent_staff(c, nodes)]
    if len(old_staffline_nodes) == 0:
        logging.info('merge_staffline_segments: nothing new to do!')
        return nodes

    canvas, (_t, _l) = draw_nodes_on_empty_canvas(old_staffline_nodes)

    _staffline_bboxes, staffline_masks = staffline_bboxes_and_masks_from_horizontal_merge(canvas)
    # Bounding boxes need to be adjusted back with respect to the original image!
    staffline_bboxes = [(t + _t, l + _l, b + _t, r + _l) for t, l, b, r in _staffline_bboxes]

    # Create the Nodes.
    next_node_id = max([c.id for c in nodes]) + 1
    dataset = nodes[0].dataset
    document = nodes[0].document

    #  - Create the staffline Nodes
    staffline_nodes = []
    for sl_bb, sl_m in zip(staffline_bboxes, staffline_masks):
        t, l, b, r = sl_bb
        c = Node(id_=next_node_id,
                 class_name=_CONST.STAFFLINE_CLASS_NAME,
                 top=t, left=l, height=b - t, width=r - l,
                 mask=sl_m,
                 dataset=dataset, document=document)
        staffline_nodes.append(c)
        next_node_id += 1

    non_staffline_nodes = [c for c in nodes
                           if c.class_name != _CONST.STAFFLINE_CLASS_NAME]
    old_staffline_ids = set([c.id for c in old_staffline_nodes])
    old2new_staffline_id_map = {}
    for os in old_staffline_nodes:
        for ns in staffline_nodes:
            if os.overlaps(ns):
                old2new_staffline_id_map[os.id] = ns

    logging.info('Re-linking from the old staffline objects to new ones.')
    for c in non_staffline_nodes:
        new_outlinks = []
        for o in c.outlinks:
            if o in old_staffline_ids:
                new_staffline = old2new_staffline_id_map[o]
                new_outlinks.append(new_staffline.id)
                new_staffline.inlinks.append(c.id)
            else:
                new_outlinks.append(o)

    output = non_staffline_nodes + staffline_nodes + already_processed_stafflines
    return output


def staffline_bboxes_and_masks_from_horizontal_merge(mask: numpy.ndarray) -> \
        Tuple[List[Tuple[int, int, int, int]], List[numpy.ndarray]]:
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
        raise ValueError('Number of stafflines is not divisible by 5!')

    return staffline_bboxes, staffline_masks


def staff_bboxes_and_masks_from_staffline_bboxes_and_image(staffline_bboxes, mask) -> \
        Tuple[List[Tuple[int, int, int, int]], List[numpy.ndarray]]:
    logging.warning('Creating staff bboxes and masks.')

    #  - Go top-down and group the stafflines by five to get staves.
    #    (The staffline bboxes are already sorted top-down.)
    staff_bboxes = []
    staff_masks = []

    n_stafflines = len(staffline_bboxes)
    for i in range(n_stafflines // 5):
        _sbb = staffline_bboxes[5 * i:5 * (i + 1)]
        _st = min([bb[0] for bb in _sbb])
        _sl = min([bb[1] for bb in _sbb])
        _sb = max([bb[2] for bb in _sbb])
        _sr = max([bb[3] for bb in _sbb])
        staff_bboxes.append((_st, _sl, _sb, _sr))
        staff_masks.append(mask[_st:_sb, _sl:_sr])

    logging.warning('Total staffs: {0}'.format(len(staff_bboxes)))

    return staff_bboxes, staff_masks


def staffline_surroundings_mask(staffline_node: Node) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Find the parts of the staffline's bounding box which lie
    above or below the actual staffline.

    These areas will be very small for straight stafflines,
    but might be considerable when staffline curvature grows.
    """
    # We segment both masks into "above staffline" and "below staffline"
    # areas.
    elevation = staffline_node.mask * 255
    # Blur, to plug small holes somewhat:
    elevation = gaussian(elevation, sigma=1.0)
    # Prepare the segmentation markers: 1 is ABOVE, 2 is BELOW
    markers = numpy.zeros(staffline_node.mask.shape)
    markers[0, :] = 1
    markers[-1, :] = 2
    markers[staffline_node.mask != 0] = 0
    seg = watershed(elevation, markers)

    bmask = numpy.ones(seg.shape)
    bmask[seg != 2] = 0
    tmask = numpy.ones(seg.shape)
    tmask[seg != 1] = 0

    return bmask, tmask


def build_staff_nodes(nodes: List[Node]) -> List[Node]:
    """Derives staff objects from staffline objects.

    Assumes each staff has 5 stafflines.

    Assumes the stafflines have already been merged."""
    stafflines = [c for c in nodes
                  if c.class_name == _CONST.STAFFLINE_CLASS_NAME and
                  not __has_parent_staff(c, nodes)]
    if len(stafflines) == 0:
        return []

    staffline_bboxes = [c.bounding_box for c in stafflines]
    canvas, (_t, _l) = draw_nodes_on_empty_canvas(stafflines)

    logging.warning('Creating staff bboxes and masks.')

    #  - Go top-down and group the stafflines by five to get staves.
    #    (The staffline bboxes are already sorted top-down.)
    staff_bboxes = []
    staff_masks = []

    n_stafflines = len(stafflines)
    for i in range(n_stafflines // 5):
        _sbb = staffline_bboxes[5 * i:5 * (i + 1)]
        _st = min([bb[0] for bb in _sbb])
        _sl = min([bb[1] for bb in _sbb])
        _sb = max([bb[2] for bb in _sbb])
        _sr = max([bb[3] for bb in _sbb])
        staff_bboxes.append((_st, _sl, _sb, _sr))
        staff_masks.append(canvas[_st - _t:_sb - _t, _sl - _l:_sr - _l])

    logging.info('Creating staff Nodes')
    next_node_id = max([c.id for c in nodes]) + 1
    dataset = nodes[0].dataset
    document = nodes[0].document

    staffs = []
    for s_bb, s_m in zip(staff_bboxes, staff_masks):
        t, l, b, r = s_bb
        staff = Node(id_=next_node_id,
                     class_name=_CONST.STAFF_CLASS_NAME,
                     top=t, left=l, height=b - t, width=r - l,
                     mask=s_m,
                     dataset=dataset, document=document)
        staffs.append(staff)
        next_node_id += 1

    for i, sc in enumerate(staffs):
        sl_from = 5 * i
        sl_to = 5 * (i + 1)
        for sl in stafflines[sl_from:sl_to]:
            sl.inlinks.append(sc.id)
            sc.outlinks.append(sl.id)

    return staffs


def build_staffspace_nodes(nodes: List[Node]) -> List[Node]:
    """Creates the staffspace objects based on stafflines
    and staffs. There is a staffspace between each two stafflines,
    one on the top side of each staff, and one on the bottom
    side for each staff (corresponding e.g. to positions of g5 and d4
    with the standard G-clef).

    Note that staffspaces do not assume anything about the number
    of stafflines per staff. However, single-staffline staffs will
    not have the outer staffspaces generated (there is nothing to derive
    their size from), for now.

    :param nodes: A list of Nodes that must contain
        all the relevant stafflines and staffs.

    :return: A list of staffspace Nodes.
    """
    next_node_id = max([c.id for c in nodes]) + 1
    dataset = nodes[0].dataset
    document = nodes[0].document

    staff_nodes = [node for node in nodes
                   if node.class_name == _CONST.STAFF_CLASS_NAME
                   and not __has_child_staffspace(node, nodes)]
    staffline_nodes = [node for node in nodes
                       if node.class_name == _CONST.STAFFLINE_CLASS_NAME
                       and not __has_neighbor_staffspace(node, nodes)]

    staff_spaces = []

    for i, staff in enumerate(staff_nodes):
        current_stafflines = [sc for sc in staffline_nodes
                              if sc.id in staff.outlinks]
        sorted_stafflines = sorted(current_stafflines, key=lambda x: x.top)

        current_staffspace_nodes = []

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
            # canvas[:s1.height, :] += s1.mask[:, dl1:s1.width-dr1]
            # canvas[-s2.height:, :] += s2.mask[:, dl2:s2.width-dr2]

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
            staffspace_mask[s1_t:s1_b, :] -= (1 - s1_below[:, dl1:s1.width - dr1])
            staffspace_mask[s2_t:s2_b, :] -= (1 - s2_above[:, dl2:s2.width - dr2])

            ss_top = s1.top
            ss_bottom = s2.bottom
            ss_left = l
            ss_right = r

            staff_space = Node(next_node_id, _CONST.STAFFSPACE_CLASS_NAME,
                               top=ss_top, left=ss_left,
                               height=ss_bottom - ss_top,
                               width=ss_right - ss_left,
                               mask=staffspace_mask,
                               dataset=dataset, document=document)

            staff_space.inlinks.append(staff.id)
            staff.outlinks.append(staff_space.id)

            current_staffspace_nodes.append(staff_space)

            next_node_id += 1

        ##########
        # Add top and bottom staffspace.
        # These outer staffspaces will have the width
        # of their bottom neighbor, and height derived
        # from its mask columns.
        # This is quite approximate, but it should do.

        # Upper staffspace
        tsl = sorted_stafflines[0]
        tsl_heights = tsl.mask.sum(axis=0)
        tss = current_staffspace_nodes[0]
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

        staff_space = Node(next_node_id, _CONST.STAFFSPACE_CLASS_NAME,
                           top=uss_top, left=uss_left,
                           height=uss_height,
                           width=uss_width,
                           mask=uss_mask,
                           dataset=dataset, document=document)
        current_staffspace_nodes.append(staff_space)
        staff.outlinks.append(staff_space.id)
        staff_space.inlinks.append(staff.id)
        next_node_id += 1

        # Lower staffspace
        bss = current_staffspace_nodes[-2]
        bss_heights = bss.mask.sum(axis=0)
        bsl = sorted_stafflines[-1]
        bsl_heights = bsl.mask.sum(axis=0)

        lss_top = bss.bottom  # + max(bsl_heights)
        lss_left = bss.left
        lss_width = bss.width
        lss_height = int(bss.height / 1.2)
        lss_mask = bss.mask[:lss_height, :] * 1

        staff_space = Node(next_node_id, _CONST.STAFFSPACE_CLASS_NAME,
                           top=lss_top, left=lss_left,
                           height=lss_height,
                           width=lss_width,
                           mask=lss_mask,
                           dataset=dataset, document=document)
        current_staffspace_nodes.append(staff_space)
        staff.outlinks.append(staff_space.id)
        staff_space.inlinks.append(staff.id)
        next_node_id += 1

        staff_spaces += current_staffspace_nodes

    return staff_spaces


def add_staff_relationships(nodes: List[Node],
                            notehead_staffspace_threshold: float = 0.2,
                            reprocess_noteheads_inside_staff_with_lls: bool = True):
    """Adds the relationships from various symbols to staff objects:
    stafflines, staffspaces, and staffs.

    :param nodes: The list of Nodes in the document. Must
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
        that are connected to leger lines, but intersect a staffline. If found,
        will remove their edges before further processing, so that the noteheads
        will seem properly unprocessed.

        Note that this handling is a bit ad-hoc for now. However, we currently
        do not have a better place to fit this in, since the Best Practices
        currently call for first applying the syntactic parser and then
        adding staff relationships.

    :return: The list of Nodes corresponding to the new graph.
    """
    graph = NotationGraph(nodes)
    id_to_node_mapping = {node.id: node for node in nodes}

    ON_STAFFLINE_RATIO_TRHESHOLD = notehead_staffspace_threshold

    ##########################################################################
    if reprocess_noteheads_inside_staff_with_lls:
        ll_noteheads_on_staff = find_noteheads_on_staff_linked_to_leger_line(nodes)
        logging.info('Reprocessing noteheads that are inside a staff, but have links'
                     ' to leger lines. Found {0} such noteheads.'
                     ''.format(len(ll_noteheads_on_staff)))
        for n in ll_noteheads_on_staff:
            # Remove all links to leger lines.
            leger_lines = graph.children(n, class_filter=[_CONST.LEGER_LINE_CLASS_NAME])
            for ll in leger_lines:
                graph.remove_edge(n.id, ll.id)

    ##########################################################################
    logging.info('Find the staff-related symbols')
    staffs = [c for c in nodes if c.class_name == _CONST.STAFF_CLASS_NAME]

    staff_related_symbols = collections.defaultdict(list)
    notehead_symbols = collections.defaultdict(list)
    rest_symbols = collections.defaultdict(list)
    for node in nodes:
        if node.class_name in _CONST.STAFF_RELATED_CLASS_NAMES:
            # Check if it already has a staff link
            if not graph.has_children(node, [_CONST.STAFF_CLASS_NAME]):
                staff_related_symbols[node.class_name].append(node)
        if node.class_name in _CONST.NOTEHEAD_CLASS_NAMES:
            if not graph.has_children(node, [_CONST.STAFF_CLASS_NAME]):
                notehead_symbols[node.class_name].append(node)
        if node.class_name in _CONST.REST_CLASS_NAMES:
            if not graph.has_children(node, [_CONST.STAFF_CLASS_NAME]):
                rest_symbols[node.class_name].append(node)

    ##########################################################################
    logging.info('Adding staff relationships')
    #  - Which direction do the relationships lead in?
    #    Need to define this.
    #
    # Staff -> symbol?
    # Symbol -> staff?
    # It does not really matter, but it's more intuitive to attach symbols
    # onto a pre-existing staff. So, symbol -> staff.
    for class_name, cs in list(staff_related_symbols.items()):
        for node in cs:
            # Find the related staff. Relatedness is measured by row overlap.
            # That means we have to modify the staff bounding box to lead
            # from the leftmost to the rightmost column. This holds
            # especially for the staff_grouping symbols.
            for s in staffs:
                st, sl, sb, sr = s.bounding_box
                sl = 0
                sr = max(sr, node.right)
                if node.overlaps((st, sl, sb, sr)):
                    link_nodes(node, s, check_that_nodes_have_the_same_document=False)

    ##########################################################################
    logging.info('Adding rest --> staff relationships.')
    for class_name, cs in list(rest_symbols.items()):
        for node in cs:
            closest_staff = min([s for s in staffs],
                                key=lambda x: ((x.bottom + x.top) / 2. - (
                                        node.bottom + node.top) / 2.) ** 2)
            link_nodes(node, closest_staff, check_that_nodes_have_the_same_document=False)

    ##########################################################################
    logging.info('Adding notehead relationships.')

    # NOTE:
    # This part should NOT rely on staffspace masks in any way!
    # They are highly unreliable.

    # Sort the staff objects top-down. Assumes stafflines do not cross,
    # and that there are no crazy curves at the end that would make the lower
    # stafflines stick out over the ones above them...
    stafflines = [c for c in nodes if c.class_name == _CONST.STAFFLINE_CLASS_NAME]
    stafflines = sorted(stafflines, key=lambda c: c.top)
    staffspaces = [c for c in nodes if c.class_name == _CONST.STAFFSPACE_CLASS_NAME]
    staffspaces = sorted(staffspaces, key=lambda c: c.top)
    staves = [c for c in nodes if c.class_name == _CONST.STAFF_CLASS_NAME]
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
    # If I know which staff (by id) and which index of staffline/staffspace,
    # I want to retrieve the given staffline/staffspace Node:
    _staff_and_idx2ss = collections.defaultdict(dict)
    _staff_and_idx2sl = collections.defaultdict(dict)

    # Build the indexes
    for _staff in staves:
        # Keep the top-down ordering from above:
        _s_stafflines = [_staffline for _staffline in stafflines
                         if _staff.id in _staffline.inlinks]
        _s_staffspaces = [_staffspace for _staffspace in staffspaces
                          if _staff.id in _staffspace.inlinks]
        for i, _sl in enumerate(_s_stafflines):
            _staff_per_ss_sl[_sl.id] = _staff
            _ss_sl_idx_wrt_staff[_sl.id] = i
            _staff_and_idx2sl[_staff.id][i] = _sl
            logging.debug('Staff {0}: stafflines {1}'.format(_staff.id,
                                                             _staff_and_idx2sl[_staff.id]))
        for i, _ss in enumerate(_s_staffspaces):
            _staff_per_ss_sl[_ss.id] = _staff
            _ss_sl_idx_wrt_staff[_ss.id] = i
            _staff_and_idx2ss[_staff.id][i] = _ss

    logging.debug(pprint.pformat(dict(_staff_and_idx2ss)))

    for class_name, cs in list(notehead_symbols.items()):
        for node in cs:

            ct, cl, cb, cr = node.bounding_box

            ################
            # Add relationship to given staffline or staffspace.

            # If notehead has leger lines, skip it for now.
            has_leger_line = False
            for outlink in node.outlinks:
                if id_to_node_mapping[outlink].class_name == _CONST.LEGER_LINE_CLASS_NAME:
                    has_leger_line = True
                    break

            if has_leger_line:
                # Attach to the appropriate staff:
                # meaning, staff closest to the innermost leger line.
                leger_lines = [id_to_node_mapping[o] for o in node.outlinks
                               if id_to_node_mapping[o].class_name == _CONST.LEGER_LINE_CLASS_NAME]
                # Furthest from notehead's top is innermost.
                # (If notehead is below staff and crosses a ll., one
                #  of these numbers will be negative. But that doesn't matter.)
                ll_max_dist = max(leger_lines, key=lambda leger_line: leger_line.top - node.top)
                # Find closest staff to max-dist leger ine
                staff_min_dist = min(staves,
                                     key=lambda ss: min((ll_max_dist.bottom - ss.top) ** 2,
                                                        (ll_max_dist.top - ss.bottom) ** 2))
                distance_of_closest_staff = (ll_max_dist.top + ll_max_dist.bottom) / 2 \
                                            - (staff_min_dist.top + staff_min_dist.bottom) / 2
                if numpy.abs(distance_of_closest_staff) > (50 + 0.5 * staff_min_dist.height):
                    logging.debug('Trying to join notehead with leger line to staff,'
                                  ' but the distance is larger than 50. Notehead: {0},'
                                  ' leger line: {1}, staff: {2}, distance: {3}'
                                  ''.format(node.uid, ll_max_dist.uid, staff_min_dist.id,
                                            distance_of_closest_staff))
                else:
                    link_nodes(node, staff_min_dist, check_that_nodes_have_the_same_document=False)
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

            if node.id < 10:
                logging.info('Notehead {0} ({1}): overlaps {2} stafflines'.format(node.uid, node.bounding_box,
                                                                                  len(overlapped_stafflines)))

            if len(overlapped_stafflines) == 1:
                s = overlapped_stafflines[0]
                dtop = s.top - ct
                dbottom = cb - s.bottom

                logging.info('Notehead {0}, staffline {1}: ratio {2:.2f}'
                             ''.format(node.id, s.id, min(dtop, dbottom) / max(dtop, dbottom)))
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
                    _staffline_idx_wrt_staff = _ss_sl_idx_wrt_staff[s.id]
                    if _is_staffspace_above:
                        _staffspace_idx_wrt_staff = _staffline_idx_wrt_staff
                    else:
                        _staffspace_idx_wrt_staff = _staffline_idx_wrt_staff + 1

                    # Retrieve the given staffsapce
                    _staff = _staff_per_ss_sl[s.id]
                    tgt_staffspace = _staff_and_idx2ss[_staff.id][_staffspace_idx_wrt_staff]
                    # Link to staffspace
                    link_nodes(node, tgt_staffspace, check_that_nodes_have_the_same_document=False)
                    # And link to staff
                    _c_staff = _staff_per_ss_sl[tgt_staffspace.id]
                    link_nodes(node, _c_staff, check_that_nodes_have_the_same_document=False)

                else:
                    # Staffline!
                    link_nodes(node, s, check_that_nodes_have_the_same_document=False)
                    # And staff:
                    _c_staff = _staff_per_ss_sl[s.id]
                    link_nodes(node, _c_staff, check_that_nodes_have_the_same_document=False)
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
                    if s.top <= node.top <= s.bottom:
                        overlapped_staffspaces[_ss_i] = min(s.bottom, node.bottom) - node.top
                    elif node.top <= s.top <= node.bottom:
                        overlapped_staffspaces[_ss_i] = s.bottom - max(node.top, s.top)

                if len(overlapped_staffspaces) == 0:
                    logging.warning('Notehead {0}: no overlapped staffline object, no leger line!'
                                    ' Expecting it will be attached to leger line and staff later on.'
                                    ''.format(node.uid))
                    continue

                _ss_i_max = max(list(overlapped_staffspaces.keys()),
                                key=lambda x: overlapped_staffspaces[x])
                max_overlap_staffspace = staffspaces[_ss_i_max]
                link_nodes(node, max_overlap_staffspace, check_that_nodes_have_the_same_document=False)
                _c_staff = _staff_per_ss_sl[max_overlap_staffspace.id]
                link_nodes(node, _c_staff, check_that_nodes_have_the_same_document=False)

            elif len(overlapped_stafflines) == 2:
                # Staffspace between those two lines.
                s1 = overlapped_stafflines[0]
                s2 = overlapped_stafflines[1]

                _staff1 = _staff_per_ss_sl[s1.id]
                _staff2 = _staff_per_ss_sl[s2.id]
                if _staff1.id != _staff2.id:
                    raise ValueError('Really weird notehead overlapping two stafflines'
                                     ' from two different staves: {0}'.format(node.uid))

                _staffspace_idx = _ss_sl_idx_wrt_staff[s2.id]
                s = _staff_and_idx2ss[_staff2.id][_staffspace_idx]
                link_nodes(node, s, check_that_nodes_have_the_same_document=False)
                # And link to staff:
                _c_staff = _staff_per_ss_sl[s.id]
                link_nodes(node, _c_staff, check_that_nodes_have_the_same_document=False)

            elif len(overlapped_stafflines) > 2:
                logging.warning('Really weird notehead overlapping more than 2 stafflines:'
                                ' {0} (permissive: linking to middle staffline)'.format(node.uid))
                # use the middle staffline -- this will be an error anyway,
                # but we want to export some MIDI more or less no matter what
                s_middle = overlapped_stafflines[len(overlapped_stafflines) // 2]
                link_nodes(node, s_middle, check_that_nodes_have_the_same_document=False)
                # And staff:
                _c_staff = _staff_per_ss_sl[s_middle.id]
                link_nodes(node, _c_staff, check_that_nodes_have_the_same_document=False)

    return nodes
