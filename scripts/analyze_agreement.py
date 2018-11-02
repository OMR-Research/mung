#!/usr/bin/env python
"""``analyze_agreement.py`` is a script that analyzes the agreement between two
annotations of the same file. The script measures:

* Object counts: are they the same?
* Object assignment: given the least-squares mapping of objects
  onto each other, to what extent do they differ?

For an overview of command-line options, call::

  analyze_agreement.py -h

Alignment algorithm
-------------------

The script uses a greedy alignment procedure.

First, it computes for each ``(truth, prediction)`` symbol pair
their recall, precision, and f-score over pixels that fall within
the mask (bounding box overlap may be misleading, mainly for
parallel beams).

Each predicted symbol is then aligned to the ground truth symbol
with the highest f-score. If the symbol classes of a ``(truth, prediction)``
pair do not match, their score gets set to 0. (This can be turned
off using the ``--no_strict_clsnames`` option.)

Next, the alignment is cleaned up: if multiple predictions are
aligned to a single ground truth, the one with the highest f-score
is chosen and the other predicted symbols are considered
unaligned.

Computing the output f-score
----------------------------

Finally, we sum all the f-scores of ``(truth, prediction)``
symbol pairs in the alignment.

Ground truth symbols that are not aligned to any predicted object
also contribute a zero to the overall f-score.

"""
from __future__ import print_function, unicode_literals
from __future__ import division
import argparse
import collections
import logging
import pprint
import time

import numpy

from mung.io import parse_cropobject_list

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


def bbox_intersection(origin, intersect):
    """Returns the coordinates of the origin bounding box that
    are intersected by the intersect bounding box.

    >>> bounding_box = 10, 100, 30, 110
    >>> other_bbox = 20, 100, 40, 105
    >>> bbox_intersection(bounding_box, other_bbox)
    (10, 0, 20, 5)
    >>> bbox_intersection(other_bbox, bounding_box)
    (0, 0, 10, 5)
    >>> containing_bbox = 4, 55, 44, 115
    >>> bbox_intersection(bounding_box, containing_bbox)
    (0, 0, 20, 10)
    >>> contained_bbox = 12, 102, 22, 108
    >>> bbox_intersection(bounding_box, contained_bbox)
    (2, 2, 12, 8)
    >>> non_overlapping_bbox = 0, 0, 3, 3
    >>> bbox_intersection(bounding_box, non_overlapping_bbox) is None
    True

    """
    o_t, o_l, o_b, o_r = origin
    t, l, b, r = intersect

    out_top = max(t, o_t)
    out_left = max(l, o_l)
    out_bottom = min(b, o_b)
    out_right = min(r, o_r)

    if (out_top < out_bottom) and (out_left < out_right):
        return out_top - o_t, \
               out_left - o_l, \
               out_bottom - o_t, \
               out_right - o_l
    else:
        return None


##############################################################################


def pixel_metrics(truth, prediction):
    """Computes the recall, precision and f-score for the prediction
    Node given the truth Node."""
    recall, precision, fscore = 0, 0, 0

    intersection_truth = bbox_intersection(truth.bounding_box,
                                           prediction.bounding_box)
    if intersection_truth is None:
        logging.debug('No intersection for CropObjects: t={0},'
                      ' p={1}'.format(truth.bounding_box,
                                      prediction.bounding_box))
        return recall, precision, fscore

    intersection_pred = bbox_intersection(prediction.bounding_box,
                                          truth.bounding_box)

    logging.debug('Found intersection for CropObjects: t={0},'
                  ' p={1}'.format(truth.bounding_box,
                                  prediction.bounding_box))

    tt, tl, tb, tr = intersection_truth
    pt, pl, pb, pr = intersection_pred
    crop_truth = truth.mask[tt:tb, tl:tr]
    crop_pred = prediction.mask[pt:pb, pl:pr]

    # Assumes the mask values are 1...

    n_truth = float(truth.mask.sum())
    n_pred = float(prediction.mask.sum())
    n_common = float((crop_truth * crop_pred).sum())

    # There are no zero-pixel objects, but the overlap may be nonzero
    if n_truth == 0:
        recall = 0.0
        precision = 0.0
    elif n_pred == 0:
        recall = 0.0
        precision = 0.0
    else:
        recall = n_common / n_truth
        precision = n_common / n_pred

    if (recall == 0) or (precision == 0):
        fscore = 0
    else:
        fscore = 2 * recall * precision / (recall + precision)

    return recall, precision, fscore


def cropobjects_rpf(truth, prediction):
    """Computes Node pixel-level metrics.

    :param truth: A list of the ground truth CropObjects.

    :param prediction: A list of the predicted CropObjects.

    :returns: Three matrices with shape ``(len(truth), len(prediction)``:
        recall, precision, and f-score for each truth/prediction Node
        pair. Truth cropobjects are rows, prediction columns.
    """
    recall = numpy.zeros((len(truth), len(prediction)))
    precision = numpy.zeros((len(truth), len(prediction)))
    fscore = numpy.zeros((len(truth), len(prediction)))

    for i, t in enumerate(truth):
        for j, p in enumerate(prediction):
            r, p, f = pixel_metrics(t, p)
            recall[i, j] = r
            precision[i, j] = p
            fscore[i, j] = f

    return recall, precision, fscore


def align_cropobjects(truth, prediction, fscore=None):
    """Aligns prediction CropObjects to truth.

    :param truth: A list of the ground truth CropObjects.

    :param prediction: A list of the predicted CropObjects.

    :returns: A list of (t, p) pairs of Node indices into
        the truth and prediction lists. There will be one
        pair for each predicted symbol.
    """
    if fscore is None:
        _, _, fscore = cropobjects_rpf(truth, prediction)

    # For each prediction (column), pick the highest-scoring
    # True symbol.
    closest_truths = list(fscore.argmax(axis=0))

    # Checking for duplicate "best ground truth" alignments.
    # This does *not* check for duplicates in the sense
    # "multiple predictions aligned to the same truth"
    # or "multiple truths aligned to the same prediction",
    # it only acts as a tie-breaker in case one prediction overlaps
    # to the same degree multiple truth objects (e.g. a single sharp
    # in a key signature).
    closest_truth_distance = [fscore[ct, j]
                              for j, ct in enumerate(closest_truths)]
    equidistant_closest_truths = [[i for i, x in enumerate(fscore[:, j])
                                   if x == ct]
                                  for j, ct in enumerate(closest_truth_distance)]

    clsname_aware_closest_truths = []
    for j, ects in enumerate(equidistant_closest_truths):
        best_truth_i = int(ects[0])

        # If there is more than one tied best choice,
        # try to choose the truth cropobject that has the same
        # class as the predicted cropobject.
        if len(ects) > 1:
            ects_c = {truth[int(i)].clsname: i for i in ects}
            j_clsname = prediction[j].clsname
            if j_clsname in ects_c:
                best_truth_i = int(ects_c[j_clsname])

        clsname_aware_closest_truths.append(best_truth_i)

    alignment = [(t, p) for p, t in enumerate(clsname_aware_closest_truths)]
    return alignment


def rpf_given_alignment(alignment, r, p,
                        n_not_aligned=0,
                        strict_clsnames=True,
                        truths=None, predictions=None):
    if strict_clsnames:
        if not truths:
            raise ValueError('If strict_clsnames is requested, must supply truths'
                             ' CropObjects!')
        if not predictions:
            raise ValueError('If strict_clsnames is requested, must supply predictions'
                             ' CropObjects!')

    total_r, total_p = 0, 0

    for i, j in alignment:

        # Check for strict clsnames only at this stage.
        # The purpose is: if two people mark the same object
        # differently, we do want to know "it should be aligned
        # to each other, but the classes don't fit" -- we don't
        # want to maybe align it to an overlapping object of
        # the corresponding wrong class.
        if strict_clsnames:
            t_c = truths[i]
            p_c = predictions[j]
            if t_c.clsname != p_c.clsname:
                continue

        total_r += r[i, j]
        total_p += p[i, j]

    # This is not correct...? What about the zeros?
    #  - Prediction with no GT is in the alignment, so it counts towards
    #    the len(alignment) denominator.
    #  - GT with no prediction aligned to it contributes zero, but is
    #    not counted towards the denominator.
    total_r /= len(alignment) + n_not_aligned
    total_p /= len(alignment) + n_not_aligned

    if (total_r == 0) or (total_p == 0):
        total_f = 0.0
    else:
        total_f = 2 * total_r * total_p / (total_r + total_p)
    return total_r, total_p, total_f


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--true', action='store', required=True,
                        help='The CropObjectList file you want to consider'
                             ' ground truth.')
    parser.add_argument('-p', '--prediction', action='store', required=True,
                        help='The CropObjectList file you want to consider'
                             ' the prediction.')

    parser.add_argument('-e', '--export', action='store',
                        help='If set, will export the problematic CropObjects'
                             ' to this file.')

    parser.add_argument('--analyze_alignment', action='store_true',
                        help='If set, will check whether the alignment is 1:1,'
                             ' and print out the irregularities.')
    parser.add_argument('--analyze_clsnames', action='store_true',
                        help='If set, will check whether the CropObjects aligned'
                             ' to each other have the same class labels'
                             ' and print out the irregularities.')
    parser.add_argument('--no_strict_clsnames', action='store_true',
                        help='If set, will not require aligned objects\' clsnames'
                             ' to match before computing pixel-wise overlap'
                             ' metrics.')
    parser.add_argument('--log_alignment', action='store_true',
                        help='Print how the true and predicted objects are'
                             ' paired.')

    parser.add_argument('--print_fscore_only', action='store_true',
                        help='If set, only print the total F-score number.'
                             ' Useful for using in an automated pipeline.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    # The algorithm:
    #  - build the cost function(s) for a pair of CropObjects
    #  - align the objects, using the cost function

    # First alignment: try just matching a predicted object to the nearest
    # true object.
    # First distance function: proportion of shared pixels.
    # Rule: if two objects don't share a pixel, they cannot be considered related.
    # Object classes do not factor into this so far.

    truth = parse_cropobject_list(args.true)
    prediction = parse_cropobject_list(args.prediction)

    _parse_time = time.clock()
    logging.info('Parsing {0} true and {1} prediction cropobjects took {2:.2f} s'
                 ''.format(len(truth), len(prediction), _parse_time - _start_time))

    r, p, f = cropobjects_rpf(truth, prediction)

    _rpf_time = time.clock()
    logging.info('Computing {0} entries of r/p/f matrices took {1:.2f} s'
                 ''.format(len(truth) * len(prediction), _rpf_time - _parse_time))

    alignment_tp = align_cropobjects(truth, prediction, fscore=f)
    alignment_pt = align_cropobjects(prediction, truth, fscore=f.T)

    # Intersect alignments
    _aln_tp_set = frozenset(alignment_tp)
    alignment_tp_symmetric = [(t, p) for p, t in alignment_pt
                              if (t, p) in _aln_tp_set
                              and (truth[t].clsname == prediction[p].clsname)]
    truth_not_aligned = [t for p, t in alignment_pt
                         if (t, p) not in alignment_tp_symmetric]
    n_truth_not_aligned = len(truth_not_aligned)
    preds_not_aligned = [p for t, p in alignment_tp
                         if (t, p) not in alignment_tp_symmetric]
    n_preds_not_aligned = len(preds_not_aligned)
    n_not_aligned = n_truth_not_aligned + n_preds_not_aligned

    _aln_time = time.clock()
    logging.info('Computing alignment took {0:.2f} s'
                 ''.format(_aln_time - _rpf_time))

    # Now compute agreement: precision and recall on pixels
    # of the aligned CropObjects.

    # We apply strict clsnames only here, after the CropObjects have been
    # aligned to each other using pixel metrics.
    _strict_clsnames = (not args.no_strict_clsnames)
    total_r, total_p, total_f = rpf_given_alignment(alignment_tp_symmetric, r, p,
                                                    n_not_aligned=n_not_aligned,
                                                    strict_clsnames=_strict_clsnames,
                                                    truths=truth,
                                                    predictions=prediction)

    if not args.print_fscore_only:
        print('Truth objs.:\t{0}'.format(len(truth)))
        print('Pred. objs.:\t{0}'.format(len(prediction)))
        print('Aligned objs.:\t{0}'.format(len(alignment_tp_symmetric)))
        print('==============================================')
        print('Recall:\t\t{0:.3f}\nPrecision:\t{1:.3f}\nF-score:\t{2:.3f}'
              ''.format(total_r, total_p, total_f))
        print('')
    else:
        print('{0:.3f}'.format(total_f))
        return

    if args.log_alignment:
        print('==============================================')
        print('Alignments:\n{0}'.format('\n'.join([
            '({0}: {1}) -- ({2}: {3})'.format(truth[t].objid, truth[t].clsname,
                                              prediction[p].objid, prediction[p].clsname)
            for t, p in alignment_tp_symmetric
        ])))
        print('Truth, not aligned:\n{0}'.format('\n'.join(['({0}: {1})'.format(truth[t].objid, truth[t].clsname)
                                                           for t in truth_not_aligned])))
        print(
            'Preds, not aligned:\n{0}'.format('\n'.join(['({0}: {1})'.format(prediction[p].objid, prediction[p].clsname)
                                                         for p in preds_not_aligned])))

    ##########################################################################
    # Check if the alignment is a pairing -- find truth objects
    # with more than one prediction aligned to them.
    if args.analyze_alignment:
        t_aln_dict = collections.defaultdict(list)
        for i, j in alignment_tp_symmetric:
            t_aln_dict[i].append(prediction[j])

        multiple_truths = [truth[i] for i in t_aln_dict
                           if len(t_aln_dict[i]) > 1]
        multiple_truths_aln_dict = {t: t_aln_dict[t]
                                    for t in t_aln_dict
                                    if len(t_aln_dict[t]) > 1}

        print('Truth multi-aligned Node classes:\n{0}'
              ''.format(pprint.pformat(
            {(truth[t].objid, truth[t].clsname): [(p.objid, p.clsname)
                                                  for p in t_aln_dict[t]]
             for t in multiple_truths_aln_dict})))

    ##########################################################################
    # Check if the aligned objects have the same classes
    if args.analyze_clsnames:
        different_clsnames_pairs = []
        for i, j in alignment_tp_symmetric:
            if truth[i].clsname != prediction[j].clsname:
                different_clsnames_pairs.append((truth[i], prediction[j]))
        print('Aligned pairs with different clsnames:\n{0}'
              ''.format('\n'.join(['{0}.{1}\t{2}.{3}'
                                   ''.format(t.objid, t.clsname, p.objid, p.clsname)
                                   for t, p in different_clsnames_pairs])))

    _end_time = time.clock()
    logging.info('analyze_agreement.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
