import collections
import copy

from typing import List, Dict, Tuple, Iterable, Any

from mung.constants import InferenceEngineConstants
from mung.graph import NotationGraph
from mung.node import Node

_CONST = InferenceEngineConstants()


class MungMatcher(object):
    """The MungMatcher class takes two MuNG notation graphs
    and outputs an optimal mapping between the graphs.

    It relies on the semantics being present.
    """

    def run(self, g1: NotationGraph, g2: NotationGraph):
        """Runs the MuNG graph matching algorithm.

        Assumes noteheads have pitches, durations and onsets filled in.
        Assumes the input graphs are nearly equal. (So far, we're only
        trying to match two graphs of the same music.)

        :param g1: The first NotationGraph.

        :param g2: The second NotationGraph.

        :return: A dict of ``(n1.id, n2.id) --> weight``.
        """
        # Starts from noteheads with onset/pitch/duration.
        noteheads_1 = self.collect_fully_defined_noteheads(g1)
        noteheads_2 = self.collect_fully_defined_noteheads(g2)

        # Matching data structure: dict with id tuples as keys, weights as values.
        # Currently, weights will only be 1 (zero weights are not in the dict).
        initial_matching = self.match_fully_defined_noteheads(noteheads_1, noteheads_2)
        _matched_prev = copy.deepcopy(set(initial_matching.keys()))

        print('Initialized matching: {0} noteheads, running growth iterations.'
              ''.format(len(_matched_prev)))
        partial_matching = self.grow_iteration(g1, g2, initial_matching)
        # Add debugprints!
        print('First growth iteration done.')
        _n_iters = 1
        while _matched_prev != set(partial_matching.keys()):
            _matched_prev = set(partial_matching.keys())

            partial_matching = self.grow_iteration(g1, g2, partial_matching)
            _n_iters += 1
            print('Growth iterations: {0}'.format(_n_iters))

        return partial_matching

    def grow_iteration(self, g1: NotationGraph, g2: NotationGraph, matching: Dict[Tuple[int, int], float]) -> \
            Dict[Tuple[int, int], float]:
        """One iteration of graph isomorphism growing.
        Algorithm:

        1. Connectivity signature matching

        * For both graphs:
            * For each vertex:
                * Find connectivity signature to already matched vertices

        * For each vertex u_i in g1 with non-empty connectivity signature:
            * For each vertex v_j in g2 of the same class as u_i:
                * If their connectivity signatures match, add them to potential
                  matches pool. (Will then resolve ties.)

        2. Within pools that share connectivity signatures:

        This is what we want to resolve with EMD. Note that these groups will be
        fairly small - at most 4 or 5 flags/beams or leger lines.

        """
        # TODO: factorize compatibility sets by object class!
        anchor_mapping_1_to_2 = {k[0]: k[1] for k in matching.keys()}

        csigs_1 = self.compute_connectivity_signatures(g1, anchors=[g1[k[0]] for k in matching.keys()])
        csigs_2 = self.compute_connectivity_signatures(g2, anchors=[g2[k[1]] for k in matching.keys()])

        # csigs contains for each vertex with a non-empty connectivity signature
        # to the current anchors the list of anchor point outlinks and inlinks.

        inverse_csigs_1 = collections.defaultdict(list)
        for v1_objid in csigs_1:
            inverse_csigs_1[csigs_1[v1_objid]].append(v1_objid)

        inverse_csigs_2 = collections.defaultdict(list)
        for v2_objid in csigs_2:
            inverse_csigs_2[csigs_2[v2_objid]].append(v2_objid)

        # We will keep the index of match-able vertices indexed
        # by their connectivity signature w.r.t g2, so we project
        # the connectivity signature from g1 through the anchor
        # mapping.

        # The compatible sets are a dict keyed by the corresponding csig_2.
        # The values are tuples of lists: first list for a given csig_2
        # contains the compatible vertices from graph 1, the second list
        # contains compatible vertices from graph 2.
        compatible_sets = collections.defaultdict(tuple)

        for v1_objid, csig_1 in csigs_1.items():
            inlinks, outlinks = csig_1
            iso_inlinks = tuple([anchor_mapping_1_to_2[i] for i in inlinks])
            iso_outlinks = tuple([anchor_mapping_1_to_2[i] for i in outlinks])
            iso_csig = (iso_inlinks, iso_outlinks)

            if iso_csig in inverse_csigs_2:
                if iso_csig not in compatible_sets:
                    compatible_sets[iso_csig] = [[], []]

                compatible_sets[iso_csig][0].append(v1_objid)
                for v2_objid in inverse_csigs_2[iso_csig]:
                    if v2_objid not in compatible_sets[iso_csig][1]:
                        compatible_sets[iso_csig][1].append(v2_objid)

        output_matching = copy.deepcopy(matching)

        # Now that we have the compatible sets, we resolve those that are
        # larger than one.
        for csig_2 in compatible_sets:
            compatible_set_objids = compatible_sets[csig_2]
            compatible_vertices_1 = [g1[node_id]
                                     for node_id in compatible_set_objids[0]]
            compatible_vertices_2 = [g2[node_id]
                                     for node_id in compatible_set_objids[1]]
            cset_matching = self.resolve_compatible_set_matching(compatible_vertices_1, compatible_vertices_2)
            # Update output matching dict
            output_matching.update(cset_matching)

        return output_matching

    def resolve_compatible_set_matching(self, vs1: List[Node], vs2: List[Node]) -> Dict[Tuple[int, int], float]:
        """Given two sets of vertices that have isomorphic
        connectivity signatures, resolves how they should actually
        be matched against each other.

        :param vs1: A list of Nodes.

        :param vs2: A list of Nodes.

        :return: A matching of these two Node lists. The matching
            is a dict of ``(n1.id, n2.id) --> weight``.
        """
        output = {}

        class_names = set([c.class_name for c in vs1 + vs2])

        for class_name in class_names:
            vs1_by_class = [c for c in vs1 if c.class_name == class_name]
            vs2_by_class = [c for c in vs2 if c.class_name == class_name]

            for v1, v2 in zip(sorted(vs1_by_class, key=lambda x: x.top),
                              sorted(vs2_by_class, key=lambda y: y.top)):
                output[(v1.id, v2.id)] = 1.0
        return output

    def compute_connectivity_signatures(self, g: NotationGraph, anchors: Iterable[Node]) -> \
            Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Computes connectivity signatures for vertices in ``g`` to the given
        ``anchors``.

        :returns: Connectivity signature dict: ``v.id --> (inlinks_tuple, outlinks_tuple)``
        """
        output = dict()
        anchor_set = set(anchors)
        for c in g.vertices:
            if c.id in anchor_set:
                continue
            anchor_outlinks = tuple([o for o in c.outlinks if o in anchor_set])
            anchor_inlinks = tuple([i for i in c.inlinks if i in anchor_set])
            if len(anchor_inlinks) + len(anchor_outlinks) > 0:
                csig = (anchor_inlinks, anchor_outlinks)
                output[c.id] = csig
        return output

    def collect_fully_defined_noteheads(self, graph: NotationGraph) -> List[Node]:
        return [c for c in graph.vertices if self.__is_fully_defined_notehead(c)]

    @staticmethod
    def __is_fully_defined_notehead(node: Node) -> bool:
        isdef = False
        if (node.class_name in _CONST.NONGRACE_NOTEHEAD_CLASS_NAMES) \
                and ('onset_beats' in node.data) \
                and ('midi_pitch_code' in node.data):
            isdef = True
        return isdef

    @staticmethod
    def _notehead_signature(c: Node) -> Tuple[Any, Any, Any]:
        return c.data['onset_beats'], c.data['midi_pitch_code'], c.data['duration_beats']

    def match_fully_defined_noteheads(self, noteheads_1: List[Node], noteheads_2: List[Node]) -> \
            Dict[Tuple[int, int], float]:
        """Matches two lists of fully defined noteheads. Only matches those
        that exactly share pitch, duration, and onset.

        :returns: A matching: dict of ``(n1.id, n2.id) --> weight``.
        """
        output = {}
        if (len(noteheads_1) == 0) or (len(noteheads_2) == 0):
            return output

        # Sort & slide algorithm makes this linear, not quadratic.
        # With more than 500 noteheads, this starts to be felt.
        ns1 = sorted(noteheads_1, key=lambda x: self._notehead_signature(x))
        ns2 = sorted(noteheads_2, key=lambda x: self._notehead_signature(x))

        j = 0
        for n1 in ns1:
            onset_1 = n1.data['onset_beats']

            n2 = ns2[j]
            # Make ns2 pointer catch up with the current note
            while onset_1 > n2.data['onset_beats']:
                j += 1
                if j >= len(ns2):
                    # We will not match anything anymore: noteheads_2 are exhausted.
                    return output
                n2 = ns2[j]

            if onset_1 == n2.data['onset_beats']:
                if self._notehead_signature(n1) == self._notehead_signature(n2):
                    output[(n1.id, n2.id)] = 1.0
                    j += 1
                    continue

            elif onset_1 < n2.data['onset_beats']:
                continue

        return output


if __name__ == '__main__':
    import os
    from mung.io import read_nodes_from_file
    from mung.graph import NotationGraph, NotationGraph, NotationGraph, NotationGraph

    test_data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'test',
                                  'test_data',
                                  'mungmatcher')
    gt_root = os.path.join(test_data_root, 'gt')
    de_root = os.path.join(test_data_root, 'detected')
    wc_root = os.path.join(test_data_root, 'without_contained')

    names = ['minifull.xml',
             'mini2full.xml']

    gt = NotationGraph(read_nodes_from_file(os.path.join(test_data_root, names[0])))
    wc = NotationGraph(read_nodes_from_file(os.path.join(test_data_root, names[1])))

    matcher = MungMatcher()

    aln = matcher.run(gt, gt)
    print('Matched GT against GT: {} gt, {} gt, {} matched'
          ''.format(len(gt), len(gt), len(aln)))

    aln = matcher.run(gt, wc)
    print('Matched GT against WC: {} gt, {} wc, {} matched'
          ''.format(len(gt), len(wc), len(aln)))

    aln = matcher.run(gt, gt)
    print('Matched WC against WC: {} wc, {} wc, {} matched'
          ''.format(len(wc), len(wc), len(aln)))
