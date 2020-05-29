"""This is a script that processes a set of symbols in order to obtain the pitch
recognition baseline. Intended to be used on top of an object detection stage."""
import argparse
import collections
import logging
import os
import pickle
import pprint
import time
import traceback

import numpy
from sklearn.feature_extraction import DictVectorizer
from typing import List, Dict, Tuple

from mung.grammar import DependencyGrammar
from mung.graph import find_beams_incoherent_with_stems, NotationGraph
from mung.graph import find_contained_nodes, remove_contained_nodes
from mung.graph import find_misdirected_leger_line_edges
from mung2midi.inference import OnsetsInferenceEngine, MIDIBuilder, PitchInferenceEngine
from mung.constants import InferenceEngineConstants as _CONST
from mung.io import parse_node_classes, read_nodes_from_file, export_node_list
from mung.node import bounding_box_intersection, merge_multiple_nodes, link_nodes, Node
from mung.stafflines import merge_staffline_segments, build_staff_nodes, \
    build_staffspace_nodes, \
    add_staff_relationships


def add_key_signatures(nodes: List[Node]) -> List[Node]:
    """Heuristic for deciding which accidentals are inline,
    and which should be interpreted as components of a key signature.

    Assumes staffline relationships have already been inferred.

    The heuristic is defined for each staff S as the following:

    * Take the leftmost clef C.
    * Take the leftmost notehead (incl. grace notes) N.
    * Take all accidentals A_S that overlap the space between C and N
      horizontally (including C, not including N), and overlap the staff.
    * Order A_S left-to-right
    * Set m = C
    * Initialize key signature K_S = {}
    * For each a_S in A_S:
    * if it is closer to m than to N, then:
    *   add a_S to K_S,
    *   set m = a_S
    * else:
    *   break

    Note that this modifies the accidentals and staffs in-place: they get inlinks
    from their respective key signatures.
    """
    graph = NotationGraph(nodes)

    new_node_id = max([m.id for m in nodes]) + 1

    key_signatures = []

    staffs = [m for m in nodes if m.class_name == _CONST.STAFF]
    for s in staffs:

        # Take the leftmost clef C.
        clefs = graph.parents(s.id, class_filter=_CONST.CLEF_CLASS_NAMES)
        if len(clefs) == 0:
            continue
        leftmost_clef = min(clefs, key=lambda x: x.left)

        # Take the leftmost notehead (incl. grace notes) N.
        noteheads = graph.parents(s.id, class_filter=_CONST.NOTEHEAD_CLASS_NAMES)
        if len(noteheads) == 0:
            continue
        leftmost_notehead = min(noteheads, key=lambda x: x.left)

        # Take all accidentals A_S that fall between C and N
        # horizontally, and overlap the staff.
        all_accidentals = [m for m in nodes
                           if m.class_name in _CONST.ACCIDENTAL_CLASS_NAMES]
        relevant_acc_bbox = s.top, leftmost_clef.left, s.bottom, leftmost_notehead.left
        relevant_accidentals = [m for m in all_accidentals
                                if bounding_box_intersection(m.bounding_box,
                                                             relevant_acc_bbox)
                                is not None]

        # Order A_S left-to-right.
        ordered_accidentals = sorted(relevant_accidentals,
                                     key=lambda x: x.left)

        # Set m = C
        current_lstop = leftmost_clef.right
        key_signature_accidentals = []

        # Iterate over accidentals; check if they are closer to lstop
        # than to the first notehead
        for a in ordered_accidentals:
            if (a.left - current_lstop) < (leftmost_notehead.left - a.right):
                key_signature_accidentals.append(a)
                current_lstop = a.right
            else:
                break

        # Build key signature and connect it to staff
        if len(key_signature_accidentals) > 0:
            key_signature_class_names = list(_CONST.KEY_SIGNATURE)[0]
            # Note: there might be spurious links from the accidentals
            # to notheads, if they were mis-interpreted during parsing.
            # This actually might not matter; needs testing.
            key_signature = merge_multiple_nodes(
                key_signature_accidentals,
                class_name=key_signature_class_names,
                id_=new_node_id)
            new_node_id += 1
            link_nodes(key_signature, s)
            for a in key_signature_accidentals:
                link_nodes(key_signature, a)

            key_signatures.append(key_signature)

    logging.info('Adding {} key signatures'.format(len(key_signatures)))
    return nodes + key_signatures


##############################################################################
# Feature extraction

class PairwiseClassificationFeatureExtractor(object):
    def __init__(self, vectorizer=None):
        """Initialize the feature extractor.

        :param vectorizer: A DictVectorizer() from scikit-learn.
            Used to convert feature dicts to the vectors that
            the edge classifier of the parser will expect.
            If None, will create a new DictVectorizer. (This is useful
            for training; you can then pickle the entire extractor
            and make sure the feature extraction works for the classifier
            at runtime.)
        """
        if vectorizer is None:
            vectorizer = DictVectorizer()
        self.vectorizer = vectorizer

    def __call__(self, *args, **kwargs):
        """The call is per item (in this case, Node pair)."""
        fd = self.get_features_relative_bbox_and_clsname(*args, **kwargs)
        # Compensate for the vecotrizer "target", which we don't have here (by :-1)
        item_features = self.vectorizer.transform(fd).toarray()[0, :-1]
        return item_features

    def get_features_relative_bbox_and_clsname(self, c_from: Node, c_to: Node):
        """Extract a feature vector from the given pair of Nodes.
        Does *NOT* convert the class names to integers.

        Features: bbox(c_to) - bbox(c_from), class_name(c_from), class_name(c_to)
        Target: 1 if there is a link from u to v

        Returns a dict that works as input to ``self.vectorizer``.
        """
        target = 0
        if c_from.document == c_to.document:
            if c_to.id in c_from.outlinks:
                target = 1
        features = (c_to.top - c_from.top,
                    c_to.left - c_from.left,
                    c_to.bottom - c_from.bottom,
                    c_to.right - c_from.right,
                    c_from.class_name,
                    c_to.class_name,
                    target)
        dt, dl, db, dr, cu, cv, tgt = features
        # Normalizing clsnames
        if cu.startswith('letter'):
            cu = 'letter'
        if cu.startswith('numeral'):
            cu = 'numeral'
        if cv.startswith('letter'):
            cv = 'letter'
        if cv.startswith('numeral'):
            cv = 'numeral'
        feature_dict = {'dt': dt,
                        'dl': dl,
                        'db': db,
                        'dr': dr,
                        'cls_from': cu,
                        'cls_to': cv,
                        'target': tgt}
        return feature_dict

    def get_features_distance_relative_bbox_and_clsname(self, from_node: Node,
                                                        to_node: Node) -> Dict:
        """Extract a feature vector from the given pair of Nodes.
        Does *NOT* convert the class names to integers.

        Features: bbox(c_to) - bbox(c_from), class_name(c_from), class_name(c_to)
        Target: 1 if there is a link from u to v

        Returns a tuple.
        """
        target = 0
        if from_node.document == to_node.document:
            if to_node.id in from_node.outlinks:
                target = 1
        distance = from_node.distance_to(to_node)
        features = (distance,
                    to_node.top - from_node.top,
                    to_node.left - from_node.left,
                    to_node.bottom - from_node.bottom,
                    to_node.right - from_node.right,
                    from_node.class_name,
                    to_node.class_name,
                    target)
        dist, dt, dl, db, dr, cu, cv, tgt = features
        if cu.startswith('letter'):
            cu = 'letter'
        if cu.startswith('numeral'):
            cu = 'numeral'
        if cv.startswith('letter'):
            cv = 'letter'
        if cv.startswith('numeral'):
            cv = 'numeral'
        feature_dict = {'dist': dist,
                        'dt': dt,
                        'dl': dl,
                        'db': db,
                        'dr': dr,
                        'cls_from': cu,
                        'cls_to': cv,
                        'target': tgt}
        return feature_dict


##############################################################################
# Edge classifier


class PairwiseClassificationParser(object):
    """This parser applies a simple classifier that takes the bounding
    boxes of two Nodes and their classes and returns whether there
    is an edge or not."""
    MAXIMUM_DISTANCE_THRESHOLD = 200

    def __init__(self, grammar: DependencyGrammar, classifier,
                 feature_extractor: PairwiseClassificationFeatureExtractor):
        self.grammar = grammar
        self.classifier = classifier
        self.extractor = feature_extractor

    def parse(self, nodes: List[Node]):

        # Ensure the same docname for all Nodes,
        # since we later compute their distances.
        # The correct docname gets set on export anyway.
        pairs, features = self.extract_all_pairs(nodes)

        logging.info(
            'Clf.Parse: {0} object pairs from {1} objects'.format(len(pairs), len(nodes)))

        preds = self.classifier.predict(features)

        edges = []
        for idx, (c_from, c_to) in enumerate(pairs):
            if preds[idx] != 0:
                edges.append((c_from.id, c_to.id))

        edges = self.__apply_trivial_fixes(nodes, edges)
        return edges

    def __apply_trivial_fixes(self, nodes: List[Node], edges: List[Tuple[int, int]]):
        edges = self.__only_one_stem_per_notehead(nodes, edges)
        edges = self.__every_full_notehead_has_a_stem(nodes, edges)

        return edges

    def __only_one_stem_per_notehead(self, nodes: List[Node], edges: List[Tuple[int, int]]):
        node_id_to_node_mapping = {n.id: n for n in nodes}  # type: Dict[int, Node]

        # Collect stems per notehead
        stems_per_notehead = collections.defaultdict(list)
        stem_objids = set()
        for from_id, to_id in edges:
            from_node = node_id_to_node_mapping[from_id]
            to_node = node_id_to_node_mapping[to_id]
            if (from_node.class_name in _CONST.NOTEHEAD_CLASS_NAMES) and \
                    (to_node.class_name == 'stem'):
                stems_per_notehead[from_id].append(to_id)
                stem_objids.add(to_id)

        # Pick the closest one (by minimum distance)
        closest_stems_per_notehead = dict()
        for n_objid in stems_per_notehead:
            n = node_id_to_node_mapping[n_objid]
            stems = [node_id_to_node_mapping[objid] for objid in stems_per_notehead[n_objid]]
            closest_stem = min(stems, key=lambda s: n.distance_to(s))
            closest_stems_per_notehead[n_objid] = closest_stem.id

        # Filter the edges
        edges = [(f_objid, t_objid) for f_objid, t_objid in edges
                 if (f_objid not in closest_stems_per_notehead) or
                 (t_objid not in stem_objids) or
                 (closest_stems_per_notehead[f_objid] == t_objid)]

        return edges

    def __every_full_notehead_has_a_stem(self, nodes: List[Node], edges):
        node_id_to_node_mapping = {c.id: c for c in nodes}  # type: Dict[int, Node]

        # Collect stems per notehead
        notehead_objids = set([c.id for c in nodes if c.class_name == 'noteheadFull'])
        stem_objids = set([c.id for c in nodes if c.class_name == 'stem'])

        noteheads_with_stem_objids = set()
        stems_with_notehead_objids = set()
        for f, t in edges:
            if node_id_to_node_mapping[f].class_name == 'noteheadFull':
                if node_id_to_node_mapping[t].class_name == 'stem':
                    noteheads_with_stem_objids.add(f)
                    stems_with_notehead_objids.add(t)

        noteheads_without_stems = {n: node_id_to_node_mapping[n] for n in notehead_objids
                                   if n not in noteheads_with_stem_objids}
        stems_without_noteheads = {n: node_id_to_node_mapping[n] for n in stem_objids
                                   if n not in stems_with_notehead_objids}

        # To each notehead, assign the closest stem that is not yet taken.
        closest_stem_per_notehead = {objid: min(stems_without_noteheads,
                                                key=lambda x: node_id_to_node_mapping[
                                                    x].distance_to(n))
                                     for objid, n in list(noteheads_without_stems.items())}

        # Filter edges that are too long
        _n_before_filter = len(closest_stem_per_notehead)
        closest_stem_threshold_distance = 80
        closest_stem_per_notehead = {n_objid: s_objid
                                     for n_objid, s_objid in list(closest_stem_per_notehead.items())
                                     if node_id_to_node_mapping[n_objid].distance_to(
                node_id_to_node_mapping[s_objid]) < closest_stem_threshold_distance}

        return edges + list(closest_stem_per_notehead.items())

    def extract_all_pairs(self, nodes: List[Node]):
        pairs = []
        features = []
        for u in nodes:
            for v in nodes:
                if u.id == v.id:
                    continue
                distance = u.distance_to(v)
                if distance < self.MAXIMUM_DISTANCE_THRESHOLD:
                    pairs.append((u, v))
                    f = self.extractor(u, v)
                    features.append(f)

        # logging.info('Parsing features: {0}'.format(features[0]))
        features = numpy.array(features)
        # logging.info('Parsing features: {0}/{1}'.format(features.shape, features))
        return pairs, features

    def is_edge(self, c_from, c_to) -> bool:
        features = self.extractor(c_from, c_to)
        result = self.classifier.predict(features)
        return result

    def set_grammar(self, grammar: DependencyGrammar):
        self.grammar = grammar


def do_parse(nodes: List[Node], parser: PairwiseClassificationParser) -> List[Node]:
    non_staff_nodes = [node for node in nodes if node.class_name not in _CONST.STAFF_CLASSES]
    edges = parser.parse(non_staff_nodes)

    # Add edges
    id_to_node_mapping = {node.id: node for node in nodes}
    for f, t in edges:
        cf, ct = id_to_node_mapping[f], id_to_node_mapping[t]
        if t not in cf.outlinks:
            if f not in ct.inlinks:
                cf.outlinks.append(t)
                ct.inlinks.append(f)

    return nodes


##############################################################################
# Staffline building

def process_stafflines(nodes: List[Node],
                       do_build_staffs: bool = True,
                       do_build_staffspaces: bool = True,
                       do_add_staff_relationships: bool = True) -> List[Node]:
    """Merges staffline fragments into stafflines. Can group them into staffs,
    add staffspaces, and add the various obligatory relationships of other
    objects to the staff objects. Required before attempting to export MIDI."""
    if len([c for c in nodes if c.class_name == 'staff']) > 0:
        logging.warning('Some stafflines have already been processed. Reprocessing'
                        ' is not certain to work.')

    try:
        new_nodes = merge_staffline_segments(nodes)
    except ValueError as e:
        logging.warning('Model: Staffline merge failed:\n\t\t'
                        '{0}'.format(e.message))
        raise

    try:
        if do_build_staffs:
            staffs = build_staff_nodes(new_nodes)
            new_nodes = new_nodes + staffs
    except Exception as e:
        logging.warning('Building staffline Nodes from merged segments failed:'
                        ' {0}'.format(e.message))
        raise

    try:
        if do_build_staffspaces:
            staffspaces = build_staffspace_nodes(new_nodes)
            new_nodes = new_nodes + staffspaces
    except Exception as e:
        logging.warning('Building staffspace Nodes from stafflines failed:'
                        ' {0}'.format(e.message))
        raise

    try:
        if do_add_staff_relationships:
            new_nodes = add_staff_relationships(new_nodes)
    except Exception as e:
        logging.warning('Adding staff relationships failed:'
                        ' {0}'.format(e.message))
        raise

    return new_nodes


def find_wrong_edges(nodes: List[Node], grammar: DependencyGrammar) -> List[Tuple[int, int]]:
    id_to_node_mapping = {node.id: node for node in nodes}
    graph = NotationGraph(nodes)

    incoherent_beam_pairs = find_beams_incoherent_with_stems(nodes)
    # Switching off misdirected leger lines: there is something wrong with them
    misdirected_leger_lines = find_misdirected_leger_line_edges(nodes)

    wrong_edges = [(n.id, b.id)
                   for n, b in incoherent_beam_pairs + misdirected_leger_lines]

    disallowed_symbol_class_pairs = [(f, t) for f, t in graph.edges
                                     if not grammar.validate_edge(id_to_node_mapping[f].class_name,
                                                                  id_to_node_mapping[t].class_name)]
    wrong_edges += disallowed_symbol_class_pairs
    return wrong_edges


def find_very_small_nodes(nodes: List[Node], bbox_threshold=40, mask_threshold=35) -> List[int]:
    very_small_nodes = []

    for c in nodes:
        total_masked_area = c.mask.sum()
        total_bbox_area = c.width * c.height
        if total_bbox_area < bbox_threshold:
            very_small_nodes.append(c)
        elif total_masked_area < mask_threshold:
            very_small_nodes.append(c)

    return list(set([c.id for c in very_small_nodes]))


def infer_precedence_edges(nodes: List[Node], factor_by_staff=True) -> List[Tuple[int, int]]:
    """Returns a list of (from_objid, to_objid) parirs. They
    then need to be added to the Nodes as precedence edges."""
    id_to_node_mapping = {c.id: c for c in nodes}
    relevant_class_names = set(list(_CONST.NONGRACE_NOTEHEAD_CLASS_NAMES)
                               + list(_CONST.REST_CLASS_NAMES))
    precedence_nodes = [c for c in nodes
                        if c.class_name in relevant_class_names]
    logging.info('_infer_precedence: {0} total prec. Nodes'
                 ''.format(len(precedence_nodes)))

    # Group the objects according to the staff they are related to
    # and infer precedence on these subgroups.
    if factor_by_staff:
        staffs = [c for c in nodes
                  if c.class_name == _CONST.STAFF]
        logging.info('_infer_precedence: got {0} staffs'.format(len(staffs)))
        staff_objids = {c.id: i for i, c in enumerate(staffs)}
        precedence_nodes_per_staff = [[] for _ in staffs]
        # All Nodes relevant for precedence have a relationship
        # to a staff.
        for c in precedence_nodes:
            for o in c.outlinks:
                if o in staff_objids:
                    precedence_nodes_per_staff[staff_objids[o]].append(c)

        logging.info('Precedence groups: {0}'
                     ''.format(precedence_nodes_per_staff))
        prec_edges = []
        for precedence_nodes_group in precedence_nodes_per_staff:
            group_prec_edges = infer_precedence_edges(precedence_nodes_group,
                                                      factor_by_staff=False)
            prec_edges.extend(group_prec_edges)
        return prec_edges

    if len(precedence_nodes) <= 1:
        logging.info('EdgeListView._infer_precedence: less than 2'
                     ' timed Nodes selected, no precedence'
                     ' edges to infer.')
        return []

    # Group into equivalence if noteheads share stems
    _stems_to_noteheads_map = collections.defaultdict(list)
    for c in precedence_nodes:
        for o in c.outlinks:
            if o not in id_to_node_mapping:
                logging.warning('Dangling outlink: {} --> {}'.format(c.id, o))
                continue
            c_o = id_to_node_mapping[o]
            if c_o.class_name == 'stem':
                _stems_to_noteheads_map[c_o.id].append(c.id)

    _prec_equiv_objids = []
    _stemmed_noteheads_objids = []
    for _stem_objid, _stem_notehead_objids in list(_stems_to_noteheads_map.items()):
        _stemmed_noteheads_objids = _stemmed_noteheads_objids \
                                    + _stem_notehead_objids
        _prec_equiv_objids.append(_stem_notehead_objids)
    for c in precedence_nodes:
        if c.id not in _stemmed_noteheads_objids:
            _prec_equiv_objids.append([c.id])

    equiv_objs = [[id_to_node_mapping[objid] for objid in equiv_objids]
                  for equiv_objids in _prec_equiv_objids]

    # Order the equivalence classes left to right
    sorted_equiv_objs = sorted(equiv_objs,
                               key=lambda eo: min([o.left for o in eo]))

    edges = []
    for i in range(len(sorted_equiv_objs) - 1):
        fr_objs = sorted_equiv_objs[i]
        to_objs = sorted_equiv_objs[i + 1]
        for f in fr_objs:
            for t in to_objs:
                edges.append((f.id, t.id))

    return edges


def add_precedence_edges(nodes: List[Node], edges: List[Tuple[int, int]]) -> List[Node]:
    """Adds precedence edges to Nodes."""
    # Ensure unique
    edges = set(edges)
    id_to_node_mapping = {c.id: c for c in nodes}

    for f, t in edges:
        cf, ct = id_to_node_mapping[f], id_to_node_mapping[t]

        if cf.data is None:
            cf.data = dict()
        if 'precedence_outlinks' not in cf.data:
            cf.data['precedence_outlinks'] = []
        cf.data['precedence_outlinks'].append(t)

        if ct.data is None:
            ct.data = dict()
        if 'precedence_inlinks' not in ct.data:
            ct.data['precedence_inlinks'] = []
        ct.data['precedence_inlinks'].append(f)

    return nodes


def build_midi(nodes: List[Node], selected_nodes: List[Node] = None,
               retain_pitches: bool = True,
               retain_durations: bool = True,
               retain_onsets: bool = True,
               tempo: int = 180):
    """Attempts to export a MIDI file from the current graph. Assumes that
    all the staff objects and their relations have been correctly established,
    and that the correct precedence graph is available.

    :param retain_pitches: If set, will record the pitch information
        in pitched objects.

    :param retain_durations: If set, will record the duration information
        in objects to which it applies.

    :returns: A single-track ``midiutil.MidiFile.MIDIFile`` object. It can be
        written to a stream using its ``mf.writeFile()`` method."""
    id_to_node_mapping = {c.id: c for c in nodes}

    pitch_inference_engine = PitchInferenceEngine()
    time_inference_engine = OnsetsInferenceEngine(nodes=nodes)

    try:
        logging.info('Running pitch inference.')
        pitches, pitch_names = pitch_inference_engine.infer_pitches(nodes,
                                                                    with_names=True)
    except Exception as e:
        logging.warning('Model: Pitch inference failed!')
        logging.exception(traceback.format_exc(e))
        raise

    if retain_pitches:
        for objid in pitches:
            c = id_to_node_mapping[objid]
            pitch_step, pitch_octave = pitch_names[objid]
            c.data['midi_pitch_code'] = pitches[objid]
            c.data['normalized_pitch_step'] = pitch_step
            c.data['pitch_octave'] = pitch_octave

    try:
        logging.info('Running durations inference.')
        durations = time_inference_engine.durations(nodes)
    except Exception as e:
        logging.warning('Model: Duration inference failed!')
        logging.exception(traceback.format_exc(e))
        raise

    if retain_durations:
        for objid in durations:
            c = id_to_node_mapping[objid]
            c.data['duration_beats'] = durations[objid]

    try:
        logging.info('Running onsets inference.')
        onsets = time_inference_engine.onsets(nodes)
    except Exception as e:
        logging.warning('Model: Onset inference failed!')
        logging.exception(traceback.format_exc(e))
        raise

    if retain_onsets:
        for objid in onsets:
            c = id_to_node_mapping[objid]
            c.data['onset_beats'] = onsets[objid]

    # Process ties
    durations, onsets = time_inference_engine.process_ties(nodes, durations, onsets)

    # Prepare selection subset
    if selected_nodes is None:
        selected_nodes = nodes
    ids_of_selected_nodes = [c.id for c in selected_nodes]

    # Build the MIDI data
    midi_builder = MIDIBuilder()
    mf = midi_builder.build_midi(
        pitches=pitches, durations=durations, onsets=onsets,
        selection=ids_of_selected_nodes, tempo=tempo)

    return mf


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-i', '--input_mung', action='store', required=True,
                        help='Read the input MuNG to this'
                             ' file.')
    parser.add_argument('-o', '--output_mung', action='store', required=True,
                        help='Write the resulting MuNG to this'
                             ' file.')

    parser.add_argument('--mlclasses', action='store', required=True,
                        help='Read the NodeClass list from this XML.')
    parser.add_argument('--grammar', action='store', required=True,
                        help='Read the grammar file that specifies the allowed edge node'
                             ' class pairs.')
    parser.add_argument('--parser', action='store', required=True,
                        help='Read the pickled feature extractor for parser classifier.')
    parser.add_argument('--vectorizer', action='store', required=True,
                        help='Read the pickled parser classifier.')

    parser.add_argument('--add_key_signatures', action='store_true',
                        help='Attempt to add key signatures. Algorithm is'
                             ' very basic: for each staff, gather all accidentals'
                             ' intersecting that staff. From the left, find the'
                             ' first notehead on a staff that is not contained in'
                             ' anything, and the last clef before'
                             ' this notehead. (...) [TESTING]')
    parser.add_argument('--filter_contained', action='store_true',
                        help='Remove objects that are fully contained within another.'
                             ' This should be applied *after* parsing non-staff'
                             ' edges, but *before* inferring staffs.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    ###############################################################
    # Preparation: loading the parsing apparatus

    with open(args.vectorizer) as hdl:
        vectorizer = pickle.load(hdl)
    feature_extractor = PairwiseClassificationFeatureExtractor(vectorizer=vectorizer)

    with open(args.parser) as hdl:
        classifier = pickle.load(hdl)

    mlclass_list = parse_node_classes(args.mlclasses)
    mlclasses = {m.name for m in mlclass_list}

    grammar = DependencyGrammar(grammar_filename=args.grammar, alphabet=mlclasses)

    parser = PairwiseClassificationParser(grammar=grammar,
                                          classifier=classifier,
                                          feature_extractor=feature_extractor)

    #################################################################
    logging.info('Load graph')
    nodes = read_nodes_from_file(args.input_mung)

    logging.info('Filter very small')
    very_small_nodes = find_very_small_nodes(nodes,
                                             bbox_threshold=40,
                                             mask_threshold=35)
    very_small_nodes = set(very_small_nodes)
    nodes = [c for c in nodes if c not in very_small_nodes]

    logging.info('Parsing')
    nodes = do_parse(nodes, parser=parser)

    # Filter contained here.
    if args.filter_contained:
        logging.info('Finding contained Nodes...')
        contained = find_contained_nodes(nodes,
                                         mask_threshold=0.95)
        NEVER_DISCARD_CLASSES = ['key_signature', 'time_signature']
        contained = [c for c in contained if c.class_name not in NEVER_DISCARD_CLASSES]

        _contained_counts = collections.defaultdict(int)
        for c in contained:
            _contained_counts[c.class_name] += 1
        logging.info('Found {} contained Nodes'.format(len(contained)))
        logging.info('Contained counts:\n{0}'.format(pprint.pformat(dict(_contained_counts))))
        nodes = remove_contained_nodes(nodes,
                                       contained)
        logging.info('Removed contained Nodes: {}...'.format([m.id for m in contained]))

    logging.info('Inferring staffline & staff objects, staff relationships')
    nodes = process_stafflines(nodes)

    if args.add_key_signatures:
        nodes = add_key_signatures(nodes)

    logging.info('Filter invalid edges')
    graph = NotationGraph(nodes)
    # Operatng on the graph changes the Nodes
    #  -- the graph only keeps a pointer
    wrong_edges = find_wrong_edges(nodes, grammar)
    for f, t in wrong_edges:
        graph.remove_edge(f, t)

    logging.info('Add precedence relationships, factored only by staff')
    prec_edges = infer_precedence_edges(nodes)
    nodes = add_precedence_edges(nodes, prec_edges)

    logging.info('Ensuring MIDI can be built')
    mf = build_midi(nodes,
                    retain_pitches=True,
                    retain_durations=True,
                    retain_onsets=True,
                    tempo=180)

    logging.info('Save output')
    docname = os.path.splitext(os.path.basename(args.output_mung))[0]
    xml = export_node_list(nodes,
                           document=docname,
                           dataset='FNOMR_results')
    with open(args.output_mung, 'w') as out_stream:
        out_stream.write(xml)
        out_stream.write('\n')

    _end_time = time.clock()
    logging.info('baseline_process_symbols.py done in {0:.3f} s'.format(_end_time - _start_time))


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
