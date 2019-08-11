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
off using the ``--no_strict_class_names`` option.)

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
import argparse
import collections
import logging
import pprint
import time

import numpy
from typing import List, Tuple, Optional

from mung.io import read_nodes_from_file
from mung.node import Node


def bounding_box_intersection(origin: Tuple[int, int, int, int], intersect: Tuple[int, int, int, int]) \
        -> Optional[Tuple[int, int, int, int]]:
    """Returns the coordinates of the origin bounding box that
    are intersected by the intersect bounding box.

    >>> bounding_box = 10, 100, 30, 110
    >>> other_bbox = 20, 100, 40, 105
    >>> bounding_box_intersection(bounding_box, other_bbox)
    (10, 0, 20, 5)
    >>> bounding_box_intersection(other_bbox, bounding_box)
    (0, 0, 10, 5)
    >>> containing_bbox = 4, 55, 44, 115
    >>> bounding_box_intersection(bounding_box, containing_bbox)
    (0, 0, 20, 10)
    >>> contained_bbox = 12, 102, 22, 108
    >>> bounding_box_intersection(bounding_box, contained_bbox)
    (2, 2, 12, 8)
    >>> non_overlapping_bbox = 0, 0, 3, 3
    >>> bounding_box_intersection(bounding_box, non_overlapping_bbox) is None
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


def pixel_metrics(truth: Node, prediction: Node) -> Tuple[float, float, float]:
    """Computes the recall, precision and f-score for the prediction
    Node given the truth Node."""
    recall, precision, fscore = 0, 0, 0

    intersection_truth = bounding_box_intersection(truth.bounding_box,
                                                   prediction.bounding_box)
    if intersection_truth is None:
        logging.debug('No intersection for Nodes: t={0},'
                      ' p={1}'.format(truth.bounding_box,
                                      prediction.bounding_box))
        return recall, precision, fscore

    intersection_pred = bounding_box_intersection(prediction.bounding_box,
                                                  truth.bounding_box)

    logging.debug('Found intersection for Nodes: t={0},'
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


def compute_recall_precision_fscore(truth: List[Node], prediction: List[Node]) \
        -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Computes Node pixel-level metrics.

    :param truth: A list of the ground truth Nodes.

    :param prediction: A list of the predicted Nodes.

    :returns: Three matrices with shape ``(len(truth), len(prediction)``:
        recall, precision, and f-score for each truth/prediction Node
        pair. Truth Nodes are rows, prediction columns.
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


def align_nodes(truth: List[Node], prediction: List[Node], fscore=None) -> List[Tuple[int, int]]:
    """Aligns prediction Nodes to truth.

    :param truth: A list of the ground truth Nodes.

    :param prediction: A list of the predicted Nodes.

    :returns: A list of (t, p) pairs of Node indices into
        the truth and prediction lists. There will be one
        pair for each predicted symbol.
    """
    if fscore is None:
        _, _, fscore = compute_recall_precision_fscore(truth, prediction)

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

    class_name_aware_closest_truths = []
    for j, ects in enumerate(equidistant_closest_truths):
        best_truth_i = int(ects[0])

        # If there is more than one tied best choice,
        # try to choose the truth Node that has the same
        # class as the predicted Node.
        if len(ects) > 1:
            ects_c = {truth[int(i)].class_name: i for i in ects}
            j_class_name = prediction[j].class_name
            if j_class_name in ects_c:
                best_truth_i = int(ects_c[j_class_name])

        class_name_aware_closest_truths.append(best_truth_i)

    alignment = [(t, p) for p, t in enumerate(class_name_aware_closest_truths)]
    return alignment


def compute_recall_precision_fscore_given_an_alignment(alignment: List[Tuple[int, int]], individual_recalls,
                                                       individual_precisions, n_not_aligned: int = 0,
                                                       strict_classnames: bool = True,
                                                       truths: List[Node] = None, predictions: List[Node] = None)\
        -> Tuple[float, float, float]:
    if strict_classnames:
        if not truths:
            raise ValueError('If strict_classnames is requested, must supply truths Nodes!')
        if not predictions:
            raise ValueError('If strict_classnames is requested, must supply predictions Nodes!')

    total_recall, total_precision = 0, 0

    for i, j in alignment:

        # Check for strict_classnames only at this stage.
        # The purpose is: if two people mark the same object
        # differently, we do want to know "it should be aligned
        # to each other, but the classes don't fit" -- we don't
        # want to maybe align it to an overlapping object of
        # the corresponding wrong class.
        if strict_classnames:
            t_c = truths[i]
            p_c = predictions[j]
            if t_c.class_name != p_c.class_name:
                continue

        total_recall += individual_recalls[i, j]
        total_precision += individual_precisions[i, j]

    # This is not correct...? What about the zeros?
    #  - Prediction with no GT is in the alignment, so it counts towards
    #    the len(alignment) denominator.
    #  - GT with no prediction aligned to it contributes zero, but is
    #    not counted towards the denominator.
    total_recall /= len(alignment) + n_not_aligned
    total_precision /= len(alignment) + n_not_aligned

    if (total_recall == 0) or (total_precision == 0):
        total_fscore = 0.0
    else:
        total_fscore = 2 * total_recall * total_precision / (total_recall + total_precision)
    return total_recall, total_precision, total_fscore


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--true', action='store', required=True,
                        help='The Nodes file you want to consider'
                             ' ground truth.')
    parser.add_argument('-p', '--prediction', action='store', required=True,
                        help='The Nodes file you want to consider'
                             ' the prediction.')

    parser.add_argument('-e', '--export', action='store',
                        help='If set, will export the problematic Nodes'
                             ' to this file.')

    parser.add_argument('--analyze_alignment', action='store_true',
                        help='If set, will check whether the alignment is 1:1,'
                             ' and print out the irregularities.')
    parser.add_argument('--analyze_classnames', action='store_true',
                        help='If set, will check whether the Nodes aligned'
                             ' to each other have the same class labels'
                             ' and print out the irregularities.')
    parser.add_argument('--no_strict_classnames', action='store_true',
                        help='If set, will not require aligned objects\' classnames'
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
    #  - build the cost function(s) for a pair of Nodes
    #  - align the objects, using the cost function

    # First alignment: try just matching a predicted object to the nearest
    # true object.
    # First distance function: proportion of shared pixels.
    # Rule: if two objects don't share a pixel, they cannot be considered related.
    # Object classes do not factor into this so far.

    ground_truth_nodes = read_nodes_from_file(args.true)
    predicted_nodes = read_nodes_from_file(args.prediction)

    _parse_time = time.clock()
    logging.info('Parsing {0} true and {1} prediction Nodes took {2:.2f} s'
                 ''.format(len(ground_truth_nodes), len(predicted_nodes), _parse_time - _start_time))

    recall, precision, fscore = compute_recall_precision_fscore(ground_truth_nodes, predicted_nodes)

    _rpf_time = time.clock()
    logging.info('Computing {0} entries of r/p/f matrices took {1:.2f} s'
                 ''.format(len(ground_truth_nodes) * len(predicted_nodes), _rpf_time - _parse_time))

    alignment_tp = align_nodes(ground_truth_nodes, predicted_nodes, fscore=fscore)
    alignment_pt = align_nodes(predicted_nodes, ground_truth_nodes, fscore=fscore.T)

    # Intersect alignments
    _aln_tp_set = frozenset(alignment_tp)
    alignment_tp_symmetric = [(t, p) for p, t in alignment_pt
                              if (t, p) in _aln_tp_set
                              and (ground_truth_nodes[t].class_name == predicted_nodes[p].class_name)]
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
    # of the aligned Nodes.

    # We apply strict classnames only here, after the Nodes have been
    # aligned to each other using pixel metrics.
    strict_classnames = (not args.no_strict_classnames)
    total_r, total_p, total_f = compute_recall_precision_fscore_given_an_alignment(alignment_tp_symmetric, recall, precision,
                                                                                   n_not_aligned=n_not_aligned,
                                                                                   strict_classnames=strict_classnames,
                                                                                   truths=ground_truth_nodes,
                                                                                   predictions=predicted_nodes)

    if not args.print_fscore_only:
        print('Truth objs.:\t{0}'.format(len(ground_truth_nodes)))
        print('Pred. objs.:\t{0}'.format(len(predicted_nodes)))
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
            '({0}: {1}) -- ({2}: {3})'.format(ground_truth_nodes[t].id, ground_truth_nodes[t].class_name,
                                              predicted_nodes[p].id, predicted_nodes[p].class_name)
            for t, p in alignment_tp_symmetric
        ])))
        print('Truth, not aligned:\n{0}'.format('\n'.join(['({0}: {1})'.format(ground_truth_nodes[t].id, ground_truth_nodes[t].class_name)
                                                           for t in truth_not_aligned])))
        print(
            'Preds, not aligned:\n{0}'.format('\n'.join(['({0}: {1})'.format(predicted_nodes[p].id, predicted_nodes[p].class_name)
                                                         for p in preds_not_aligned])))

    ##########################################################################
    # Check if the alignment is a pairing -- find truth objects
    # with more than one prediction aligned to them.
    if args.analyze_alignment:
        t_aln_dict = collections.defaultdict(list)
        for i, j in alignment_tp_symmetric:
            t_aln_dict[i].append(predicted_nodes[j])

        multiple_truths = [ground_truth_nodes[i] for i in t_aln_dict
                           if len(t_aln_dict[i]) > 1]
        multiple_truths_aln_dict = {t: t_aln_dict[t]
                                    for t in t_aln_dict
                                    if len(t_aln_dict[t]) > 1}

        print('Truth multi-aligned Node classes:\n{0}'
              ''.format(pprint.pformat(
            {(ground_truth_nodes[t].id, ground_truth_nodes[t].class_name): [(p.id, p.class_name)
                                                  for p in t_aln_dict[t]]
             for t in multiple_truths_aln_dict})))

    ##########################################################################
    # Check if the aligned objects have the same classes
    if args.analyze_classnames:
        different_classnames_pairs = []
        for i, j in alignment_tp_symmetric:
            if ground_truth_nodes[i].class_name != predicted_nodes[j].class_name:
                different_classnames_pairs.append((ground_truth_nodes[i], predicted_nodes[j]))
        print('Aligned pairs with different class_names:\n{0}'
              ''.format('\n'.join(['{0}.{1}\t{2}.{3}'
                                   ''.format(t.id, t.class_name, p.id, p.class_name)
                                   for t, p in different_classnames_pairs])))

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
