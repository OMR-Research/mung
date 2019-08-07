import logging

import numpy
from skimage.measure import label
from typing import Tuple, Dict, Any, List

from mung.inference.constants import InferenceEngineConstants as _CONST


def connected_components2bboxes(labels):
    """Returns a dictionary of bounding boxes (upper left c., lower right c.)
    for each label.

    >>> labels = [[0, 0, 1, 1], [2, 0, 0, 1], [2, 0, 0, 0], [0, 0, 3, 3]]
    >>> bboxes = connected_components2bboxes(labels)
    >>> bboxes[0]
    [0, 0, 4, 4]
    >>> bboxes[1]
    [0, 2, 2, 4]
    >>> bboxes[2]
    [1, 0, 3, 1]
    >>> bboxes[3]
    [3, 2, 4, 4]


    :param labels: The output of cv2.connectedComponents().

    :returns: A dict indexed by labels. The values are quadruplets
        (xmin, ymin, xmax, ymax) so that the component with the given label
        lies exactly within labels[xmin:xmax, ymin:ymax].
    """
    bboxes = {}
    for x, row in enumerate(labels):
        for y, l in enumerate(row):
            if l not in bboxes:
                bboxes[l] = [x, y, x + 1, y + 1]
            else:
                box = bboxes[l]
                if x < box[0]:
                    box[0] = x
                elif x + 1 > box[2]:
                    box[2] = x + 1
                if y < box[1]:
                    box[1] = y
                elif y + 1 > box[3]:
                    box[3] = y + 1
    return bboxes


def compute_connected_components(image: numpy.ndarray) -> \
        Tuple[int, numpy.ndarray, Dict[int, List[int]]]:
    labels = label(image, background=0)
    number_of_connected_components = int(labels.max())
    bboxes = connected_components2bboxes(labels)
    return number_of_connected_components, labels, bboxes


def resolve_notehead_wrt_staffline(notehead, staffline_or_leger_line):
    """Resolves the relative vertical position of the notehead with respect
    to the given staff_line or legerLine object. Returns -1 if notehead
    is *below* staffline, 0 if notehead is *on* staffline, and 1 if notehead
    is *above* staffline."""
    ll = staffline_or_leger_line

    # Determining whether the notehead is on a leger
    # line or in the adjacent temp staffspace.
    # This uses a magic number, ON_STAFFLINE_RATIO_THRESHOLD.
    output_position = 0

    ### DEBUG!!!
    dtop, dbottom = 1, 1

    # Weird situation with notehead vertically *inside* bbox
    # of leger line (could happen with slanted LLs and very small
    # noteheads).
    if ll.top <= notehead.top <= notehead.bottom <= ll.bottom:
        output_position = 0

    # No vertical overlap between LL and notehead
    elif ll.top > notehead.bottom:
        output_position = 1
    elif notehead.top > ll.bottom:
        output_position = -1

    # Complicated situations: overlap
    else:
        # Notehead "around" leger line.
        if notehead.top < ll.top <= ll.bottom < notehead.bottom:
            dtop = ll.top - notehead.top
            dbottom = notehead.bottom - ll.bottom

            if min(dtop, dbottom) / max(dtop, dbottom) \
                    < _CONST.ON_STAFFLINE_RATIO_THRESHOLD:
                if dtop > dbottom:
                    output_position = 1
                else:
                    output_position = -1

        # Notehead interlaced with leger line, notehead on top
        elif notehead.top < ll.top <= notehead.bottom <= ll.bottom:
            # dtop = closest_ll.top - notehead.top
            # dbottom = max(notehead.bottom - closest_ll.top, 1)
            # if float(dbottom) / float(dtop) \
            #         < InferenceEngineConstants.ON_STAFFLINE_RATIO_TRHESHOLD:
            output_position = 1

        # Notehead interlaced with leger line, ledger line on top
        elif ll.top <= notehead.top <= ll.bottom < notehead.bottom:
            # dtop = max(closest_ll.bottom - notehead.top, 1)
            # dbottom = notehead.bottom - closest_ll.bottom
            # if float(dtop) / float(dbottom) \
            #         < InferenceEngineConstants.ON_STAFFLINE_RATIO_TRHESHOLD:
            output_position = -1

        else:
            logging.warning('Strange notehead {0} vs. leger line {1}'
                         ' situation: bbox notehead {2}, LL {3}.'
                         ' Note that the output position is unusable;'
                         ' pleasre re-do this attachment manually.'
                         ''.format(notehead.uid, ll.uid,
                                   notehead.bounding_box,
                                   ll.bounding_box))
    return output_position


def is_notehead_on_line(notehead, line_obj):
    """Check whether given notehead is positioned on the line object."""
    if line_obj.clsname not in _CONST.STAFFLINE_LIKE_CLASS_NAMES:
        raise ValueError('Cannot resolve relative position of notehead'
                         ' {0} to non-staffline-like object {1}'
                         ''.format(notehead.uid, line_obj.uid))

    position = resolve_notehead_wrt_staffline(notehead, line_obj)
    return position == 0
