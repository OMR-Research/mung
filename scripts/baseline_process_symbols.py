"""This is a script that processes a set of symbols in order to obtain the pitch
recognition baseline. Intended to be used on top of an object detection stage."""
import argparse
import codecs
import collections
import logging
import os
import pickle
import pprint
import time
import traceback

import numpy
from sklearn.feature_extraction import DictVectorizer
from typing import List, Dict

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


def add_key_signatures(cropobjects):
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
    _g = NotationGraph(cropobjects)

    _current_key_signature_objid = max([m.objid for m in cropobjects]) + 1

    key_signatures = []

    staffs = [m for m in cropobjects if m.clsname == _CONST.STAFF_CLSNAME]
    for s in staffs:

        # Take the leftmost clef C.
        clefs = _g.parents(s.objid, class_filter=_CONST.CLEF_CLSNAMES)
        if len(clefs) == 0:
            continue
        leftmost_clef = min(clefs, key=lambda x: x.left)

        # Take the leftmost notehead (incl. grace notes) N.
        noteheads = _g.parents(s.objid, class_filter=_CONST.NOTEHEAD_CLSNAMES)
        if len(noteheads) == 0:
            continue
        leftmost_notehead = min(noteheads, key=lambda x: x.left)

        # Take all accidentals A_S that fall between C and N
        # horizontally, and overlap the staff.
        all_accidentals = [m for m in cropobjects
                           if m.clsname in _CONST.ACCIDENTAL_CLSNAMES]
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
            _key_signature_clsname = list(_CONST.KEY_SIGNATURE_CLSNAMES)[0]
            # Note: there might be spurious links from the accidentals
            # to notheads, if they were mis-interpreted during parsing.
            # This actually might not matter; needs testing.
            key_signature = merge_multiple_nodes(
                key_signature_accidentals,
                class_name=_key_signature_clsname,
                id_=_current_key_signature_objid)
            _current_key_signature_objid += 1
            link_nodes(key_signature, s)
            for a in key_signature_accidentals:
                link_nodes(key_signature, a)

            key_signatures.append(key_signature)

    logging.info('Adding {} key signatures'.format(len(key_signatures)))
    return cropobjects + key_signatures


##############################################################################
# Grammar: restricting allowed edges & cardinalities based on symbol classes
# Note: the Grammar and Parser classes are all copied out of MUSCIMarker!


class DependencyGrammar(object):
    """The DependencyGrammar class implements rules about valid graphs above
    objects from a set of recognized classes.

    The Grammar complements a Parser. It defines rules, and the Parser
    implements algorithms to apply these rules to some input.

    A grammar has an **Alphabet** and **Rules**. The alphabet is a list
    of symbols that the grammar recognizes. Rules are constraints on
    the structures that can be induced among these symbols.

    There are two kinds of grammars according to what kinds of rules
    they use: **dependency** rules, and **constituency** rules. Dependency
    rules specify which symbols are governing, and which symbols are governed::

      noteheadFull | stem

    There can be multiple left-hand side and right-hand side symbols,
    as a shortcut for a list of rules::

        noteheadFull | stem beam
        noteheadFull noteheadHalf | legerLine durationDot tie notehead*Small

    The asterisk works as a wildcard. Currently, only one wildcard per symbol
    is allowed::

      time_signature | numeral_*

    Lines starting with a ``#`` are regarded as comments and ignored.
    Empty lines are also ignored.


    Constituency grammars consist of *rewriting rules*, such as::

      Note -> notehead stem | notehead stem duration-dot

    Constituency grammars also distinguish between *terminal* symbols, which
    can only occur on the right-hand side of the rules, and *non-terminal*
    symbols, which can also occur on the left-hand side. They are implemented
    in the class ``ConstituencyGrammar``.

    Cardinality rules
    -----------------

    We can also specify in the grammar the minimum and/or maximum number
    of relationships, both inlinks and outlinks, that an object can form
    with other objects of given types. For example:

    * One notehead may have up to two stems attached.
    * We also allow for stemless full noteheads.
    * One stem can be attached to multiple noteheads, but at least one.

    This would be expressed as::

      ``notehead-*{,2} | stem{1,}``

    The relationship of noteheads to leger lines is generally ``m:n``::

      ``noteheadFull | legerLine``

    A time signature may consist of multiple numerals, but only one
    other symbol::

      time_signature{1,} | numeral_*{1}
      time_signature{1} | whole-time_mark alla_breve other_time_signature

    A key signature may have any number of sharps and flats.
    A sharp or flat can only belong to one key signature. However,
    not every sharp belongs to a key signature::

      keySignature | accidentalSharp{,1} accidentalFlat{,1} accidentalNatural{,1} accidentalDoubleSharp{,1} accidentalDoubleFlat{,1}

    For the left-hand side of the rule, the cardinality restrictions apply to
    outlinks towards symbols of classes on the right-hand side of the rule.
    For the right-hand side, the cardinality restrictions apply to inlinks
    from symbols of left-hand side classes.

    It is also possible to specify that regardless of where outlinks
    lead, a symbol should always have at least some::

      timeSignature{1,} |
      repeat{2,} |

    And analogously for inlinks:

      | letter*{1,}
      | numeral*{1,}
      | legerLine{1,}
      | notehead*Small{1,}

    Interface
    ---------

    The basic role of the dependency grammar is to provide the list of rules:

    >>> from mung.io import parse_node_classes
    >>> fpath = os.path.dirname(os.path.dirname(__file__)) + '/test/test_data/mff-muscima-classes-annot.deprules'
    >>> mlpath = os.path.dirname(os.path.dirname(__file__)) + '/test/test_data/mff-muscima-classes-annot.xml'
    >>> mlclass_dict = {node_class.class_id: node_class for node_class in parse_node_classes(mlpath)}
    >>> g = DependencyGrammar(grammar_filename=fpath, mlclasses=mlclass_dict)
    >>> len(g.rules)
    646

    Grammar I/O
    -----------

    The alphabet is stored by means of the already-familiar MLClassList.

    The rules are stored in *rule files*. For the grammars included
    in MUSCIMarker, rule files are stored in the ``data/grammars/``
    directory.

    A rule file line can be empty, start with a ``#`` (comment), or contain
    a rule symbol ``|``. Empty lines and comments are ignored during parsing.
    Rules are split into left- and right-hand side tokens, according to
    the position of the ``|`` symbol.

    Parsing a token returns the token string (unexpanded wildcards), its
    minimum and maximum cardinality in the rule (defaults are ``(0, 10000)``
    if no cardinality is provided).

    >>> g.parse_token('notehead-*')
    ('notehead-*', 0, 10000)
    >>> g.parse_token('notehead-*{1,5}')
    ('notehead-*', 1, 5)
    >>> g.parse_token('notehead-*{1,}')
    ('notehead-*', 1, 10000)
    >>> g.parse_token('notehead-*{,5}')
    ('notehead-*', 0, 5)
    >>> g.parse_token('notehead-*{1}')
    ('notehead-*', 1, 1)

    The wildcards are expanded at the level of a line.

    >>> l = 'notehead*{,2} | stem'
    >>> rules, inlink_cards, outlink_cards, _, _ = g.parse_dependency_grammar_line(l)
    >>> rules
    [('noteheadFull', 'stem'), ('noteheadFullSmall', 'stem'), ('noteheadHalfSmall', 'stem'), ('noteheadHalf', 'stem'), ('noteheadWhole', 'stem')]
    >>> outlink_cards['noteheadHalf']
    {'stem': (0, 2)}
    >>> inlink_cards['stem']
    {'noteheadFull': (0, 10000), 'noteheadFullSmall': (0, 10000), 'noteheadHalfSmall': (0, 10000), 'noteheadHalf': (0, 10000), 'noteheadWhole': (0, 10000)}

    A key signature can have any number of sharps, flats, or naturals,
    but if a given symbol is part of a key signature, it can only be part of one.

    >>> l = 'key-signature | sharp{1} flat{1} natural{1}'
    >>> rules, inlink_cards, _, _, _ = g.parse_dependency_grammar_line(l)
    >>> rules
    [('key-signature', 'sharp'), ('key-signature', 'flat'), ('key-signature', 'natural')]
    >>> inlink_cards
    {'sharp': {'key-signature': (1, 1)}, 'flat': {'key-signature': (1, 1)}, 'natural': {'key-signature': (1, 1)}}

    You can also give *aggregate* cardinality rules, of the style "whatever rule
    applies, there should be at least X/at most Y edges for this type of object".

    >>> l = 'keySignature{1,} |'
    >>> _, _, _, _, out_aggregate_cards = g.parse_dependency_grammar_line(l)
    >>> out_aggregate_cards
    {'keySignature': (1, 10000)}
    >>> l = 'notehead*Small{1,} |'
    >>> _, _, _, _, out_aggregate_cards = g.parse_dependency_grammar_line(l)
    >>> out_aggregate_cards
    {'noteheadFullSmall': (1, 10000), 'noteheadHalfSmall': (1, 10000)}
    >>> l = '| beam{1,} stem{1,} accidentalFlat{1,}'
    >>> _, _, _, in_aggregate_cards, _ = g.parse_dependency_grammar_line(l)
    >>> in_aggregate_cards
    {'beam': (1, 10000), 'stem': (1, 10000), 'accidentalFlat': (1, 10000)}

    """

    WILDCARD = '*'

    _MAX_CARD = 10000

    def __init__(self, grammar_filename: str, mlclasses):
        """Initialize the Grammar: fill in alphabet and parse rules."""
        self.alphabet = {str(m.name): m for m in list(mlclasses.values())}
        # logging.info('DependencyGrammar: got alphabet:\n{0}'
        #              ''.format(pprint.pformat(self.alphabet)))
        self.rules = []
        self.inlink_cardinalities = {}
        '''Keys: classes, values: dict of {from: (min, max)}'''

        self.outlink_cardinalities = {}
        '''Keys: classes, values: dict of {to: (min, max)}'''

        self.inlink_aggregated_cardinalities = {}
        '''Keys: classes, values: (min, max)'''

        self.outlink_aggregated_cardinalities = {}
        '''Keys: classes, values: (min, max)'''

        rules, ic, oc, iac, oac = self.parse_dependency_grammar_rules(grammar_filename)
        if self._validate_rules(rules):
            self.rules = rules
            logging.info('DependencyGrammar: Imported {0} rules'
                         ''.format(len(self.rules)))
            self.inlink_cardinalities = ic
            self.outlink_cardinalities = oc
            self.inlink_aggregated_cardinalities = iac
            self.outlink_aggregated_cardinalities = oac
            logging.debug('DependencyGrammar: Inlink aggregated cardinalities: {0}'
                          ''.format(pprint.pformat(iac)))
            logging.debug('DependencyGrammar: Outlink aggregated cardinalities: {0}'
                          ''.format(pprint.pformat(oac)))
        else:
            raise ValueError('Not able to parse dependency grammar file {0}.'
                             ''.format(grammar_filename))

    def validate_edge(self, head_name, child_name):
        return (head_name, child_name) in self.rules

    def validate_graph(self, vertices, edges):
        """Checks whether the given graph complies with the grammar.

        :param vertices: A dict with any keys and values corresponding
            to the alphabet of the grammar.

        :param edges: A list of ``(from, to)`` pairs, where both
            ``from`` and ``to`` are valid keys into the ``vertices`` dict.

        :returns: ``True`` if the graph is valid, ``False`` otherwise.
        """
        v, i, o = self.find_invalid_in_graph(vertices=vertices, edges=edges)
        return len(v) == 0

    def find_invalid_in_graph(self, vertices, edges, provide_reasons=False):
        """Finds vertices and edges where the given object graph does
        not comply with the grammar.

        Wrong vertices are any that:

        * are not in the alphabet;
        * have a wrong inlink or outlink;
        * have missing outlinks or inlinks.

        Discovering missing edges is difficult, because the grammar
        defines cardinalities on a per-rule basis and there is currently
        no way to make a rule compulsory, or to require at least one rule
        from a group to apply. It is currently not implemented.

        Wrong outlinks are such that:

        * connect symbol pairs that should not be connected based on their
          classes;
        * connect so that they exceed the allowed number of outlinks to
          the given symbol type

        Wrong inlinks are such that:

        * connect symbol pairs that should not be connected based on their
          classes;
        * connect so that they exceed the allowed number of inlinks
          to the given symbol based on the originating symbols' classes.

        :param vertices: A dict with any keys and values corresponding
            to the alphabet of the grammar.

        :param edges: A list of ``(from, to)`` pairs, where both
            ``from`` and ``to`` are valid keys into the ``vertices`` dict.

        :returns: A list of vertices, a list of inlinks and a list of outlinks
            that do not comply with the grammar.
        """
        logging.info('DependencyGrammar: looking for errors.')

        wrong_vertices = []
        wrong_inlinks = []
        wrong_outlinks = []

        reasons_v = {}
        reasons_i = {}
        reasons_o = {}

        # Check that vertices have labels that are in the alphabet
        for v, clsname in vertices.items():
            if clsname not in self.alphabet:
                wrong_vertices.append(v)
                reasons_v[v] = 'Symbol {0} not in alphabet: class {1}.' \
                               ''.format(v, clsname)

        # Check that all edges are allowed
        for f, t in edges:
            nf, nt = str(vertices[f]), str(vertices[t])
            if (nf, nt) not in self.rules:
                logging.warning('Wrong edge: {0} --> {1}, rules:\n{2}'
                                ''.format(nf, nt, pprint.pformat(self.rules)))

                wrong_inlinks.append((f, t))
                reasons_i[(f, t)] = 'Outlink {0} ({1}) -> {2} ({3}) not in ' \
                                    'alphabet.'.format(nf, f, nt, t)

                wrong_outlinks.append((f, t))
                reasons_o[(f, t)] = 'Outlink {0} ({1}) -> {2} ({3}) not in ' \
                                    'alphabet.'.format(nf, f, nt, t)
                if f not in wrong_vertices:
                    wrong_vertices.append(f)
                    reasons_v[f] = 'Symbol {0} (class: {1}) participates ' \
                                   'in wrong outlink: {2} ({3}) --> {4} ({5})' \
                                   ''.format(f, vertices[f], nf, f, nt, t)
                if t not in wrong_vertices:
                    wrong_vertices.append(t)
                    reasons_v[t] = 'Symbol {0} (class: {1}) participates ' \
                                   'in wrong inlink: {2} ({3}) --> {4} ({5})' \
                                   ''.format(t, vertices[t], nf, f, nt, t)

        # Check aggregate cardinality rules
        #  - build inlink and outlink dicts
        inlinks = {}
        outlinks = {}
        for v in vertices:
            outlinks[v] = set()
            inlinks[v] = set()
        for f, t in edges:
            outlinks[f].add(t)
            inlinks[t].add(f)

        # If there are not enough edges, the vertex itself is wrong
        # (and none of the existing edges are wrong).
        # Currently, if there are too many edges, the vertex itself
        # is wrong and none of the existing edges are marked.
        #
        # Future:
        # If there are too many edges, the vertex itself and *all*
        # the edges are marked as wrong (because any of them is the extra
        # edge, and it's easiest to just delete them and start parsing
        # again).
        logging.debug('DependencyGrammar: checking outlink aggregate cardinalities'
                      '\n{0}'.format(pprint.pformat(outlinks)))
        for f in outlinks:
            f_clsname = vertices[f]
            if f_clsname not in self.outlink_aggregated_cardinalities:
                # Given vertex has no aggregate cardinality restrictions
                continue
            cmin, cmax = self.outlink_aggregated_cardinalities[f_clsname]
            logging.debug('DependencyGrammar: checking outlink cardinality'
                          ' rule fulfilled for vertex {0} ({1}): should be'
                          ' within {2} -- {3}'.format(f, vertices[f], cmin, cmax))
            if not (cmin <= len(outlinks[f]) <= cmax):
                wrong_vertices.append(f)
                reasons_v[f] = 'Symbol {0} (class: {1}) has {2} outlinks,' \
                               ' but grammar specifies {3} -- {4}.' \
                               ''.format(f, vertices[f], len(outlinks[f]),
                                         cmin, cmax)

        for t in inlinks:
            t_clsname = vertices[t]
            if t_clsname not in self.inlink_aggregated_cardinalities:
                continue
            cmin, cmax = self.inlink_aggregated_cardinalities[t_clsname]
            if not (cmin <= len(inlinks[t]) <= cmax):
                wrong_vertices.append(t)
                reasons_v[t] = 'Symbol {0} (class: {1}) has {2} inlinks,' \
                               ' but grammar specifies {3} -- {4}.' \
                               ''.format(f, vertices[f], len(inlinks[f]),
                                         cmin, cmax)

        # Now check for rule-based inlinks and outlinks.
        # for f in outlinks:
        #    oc = self.outlink_cardinalities[f]
        if provide_reasons:
            return wrong_vertices, wrong_inlinks, wrong_outlinks, \
                   reasons_v, reasons_i, reasons_o

        return wrong_vertices, wrong_inlinks, wrong_outlinks

    def parse_dependency_grammar_rules(self, filename):
        """Returns the Rules stored in the given rule file."""
        rules = []
        inlink_cardinalities = {}
        outlink_cardinalities = {}

        inlink_aggregated_cardinalities = {}
        outlink_aggregated_cardinalities = {}

        _invalid_lines = []
        with codecs.open(filename, 'r', 'utf-8') as hdl:
            for line_no, line in enumerate(hdl):
                l_rules, in_card, out_card, in_agg_card, out_agg_card = self.parse_dependency_grammar_line(
                    line)

                if not self._validate_rules(l_rules):
                    _invalid_lines.append((line_no, line))

                rules.extend(l_rules)

                # Update cardinalities
                for lhs in out_card:
                    if lhs not in outlink_cardinalities:
                        outlink_cardinalities[lhs] = dict()
                    outlink_cardinalities[lhs].update(out_card[lhs])

                for rhs in in_card:
                    if rhs not in inlink_cardinalities:
                        inlink_cardinalities[rhs] = dict()
                    inlink_cardinalities[rhs].update(in_card[rhs])

                inlink_aggregated_cardinalities.update(in_agg_card)
                outlink_aggregated_cardinalities.update(out_agg_card)

        if len(_invalid_lines) > 0:
            logging.warning('DependencyGrammar.parse_rules(): Invalid lines'
                            ' {0}'.format(pprint.pformat(_invalid_lines)))

        return rules, inlink_cardinalities, outlink_cardinalities, \
               inlink_aggregated_cardinalities, outlink_aggregated_cardinalities

    def parse_dependency_grammar_line(self, line):
        """Parse one dependency grammar line. See DependencyGrammar
        I/O documentation for the format."""
        rules = []
        out_cards = {}
        in_cards = {}
        out_agg_cards = {}
        in_agg_cards = {}

        if line.strip().startswith('#'):
            return [], dict(), dict(), dict(), dict()
        if len(line.strip()) == 0:
            return [], dict(), dict(), dict(), dict()
        if '|' not in line:
            return [], dict(), dict(), dict(), dict()

        # logging.info('DependencyGrammar: parsing rule line:\n\t\t{0}'
        #              ''.format(line.rstrip('\n')))
        lhs, rhs = line.strip().split('|', 1)
        lhs_tokens = lhs.strip().split()
        rhs_tokens = rhs.strip().split()

        # logging.info('DependencyGrammar: tokens lhs={0}, rhs={1}'
        #             ''.format(lhs_tokens, rhs_tokens))

        # Normal rule line? Aggregate cardinality line?
        _line_type = 'normal'
        if len(lhs) == 0:
            _line_type = 'aggregate_inlinks'
        if len(rhs) == 0:
            _line_type = 'aggregate_outlinks'

        logging.debug('Line {0}: type {1}, lhs={2}, rhs={3}'.format(line, _line_type, lhs, rhs))

        if _line_type == 'aggregate_inlinks':
            rhs_tokens = rhs.strip().split()
            for rt in rhs_tokens:
                token, rhs_cmin, rhs_cmax = self.parse_token(rt)
                for t in self._matching_names(token):
                    in_agg_cards[t] = (rhs_cmin, rhs_cmax)
            logging.debug('DependencyGrammar: found inlinks: {0}'
                          ''.format(pprint.pformat(in_agg_cards)))
            return rules, out_cards, in_cards, in_agg_cards, out_agg_cards

        if _line_type == 'aggregate_outlinks':
            lhs_tokens = lhs.strip().split()
            for lt in lhs_tokens:
                token, lhs_cmin, lhs_cmax = self.parse_token(lhs.strip())
                for t in self._matching_names(token):
                    out_agg_cards[t] = (lhs_cmin, lhs_cmax)
            logging.debug('DependencyGrammar: found outlinks: {0}'
                          ''.format(pprint.pformat(out_agg_cards)))
            return rules, out_cards, in_cards, in_agg_cards, out_agg_cards

        # Normal line that defines a left-hand side and a right-hand side

        lhs_symbols = []
        # These cardinalities apply to all left-hand side tokens,
        # for edges leading to any of the right-hand side tokens.
        lhs_cards = {}
        for l in lhs_tokens:
            token, lhs_cmin, lhs_cmax = self.parse_token(l)
            all_tokens = self._matching_names(token)
            lhs_symbols.extend(all_tokens)
            for t in all_tokens:
                lhs_cards[t] = (lhs_cmin, lhs_cmax)

        rhs_symbols = []
        rhs_cards = {}
        for r in rhs_tokens:
            token, rhs_cmin, rhs_cmax = self.parse_token(r)
            all_tokens = self._matching_names(token)
            rhs_symbols.extend(all_tokens)
            for t in all_tokens:
                rhs_cards[t] = (rhs_cmin, rhs_cmax)

        # logging.info('DependencyGrammar: symbols lhs={0}, rhs={1}'
        #              ''.format(lhs_symbols, rhs_symbols))

        # Build the outputs from the cartesian product
        # of left-hand and right-hand tokens.
        for l in lhs_symbols:
            if l not in out_cards:
                out_cards[l] = {}
            for r in rhs_symbols:
                if r not in in_cards:
                    in_cards[r] = {}

                rules.append((l, r))
                out_cards[l][r] = lhs_cards[l]
                in_cards[r][l] = rhs_cards[r]

        # logging.info('DependencyGramamr: got rules:\n{0}'
        #              ''.format(pprint.pformat(rules)))
        # logging.info('DependencyGrammar: got inlink cardinalities:\n{0}'
        #              ''.format(pprint.pformat(in_cards)))
        # logging.info('DependencyGrammar: got outlink cardinalities:\n{0}'
        #              ''.format(pprint.pformat(out_cards)))
        return rules, in_cards, out_cards, in_agg_cards, out_agg_cards

    def parse_token(self, l):
        """Parse one *.deprules file token. See class documentation for
        examples.

        :param l: One token of a *.deprules file.

        :return: token, cmin, cmax
        """
        l = str(l)
        cmin, cmax = 0, self._MAX_CARD
        if '{' not in l:
            token = l
        else:
            token, cardinality = l[:-1].split('{')
            if ',' not in cardinality:
                cmin, cmax = int(cardinality), int(cardinality)
            else:
                cmin_string, cmax_string = cardinality.split(',')
                if len(cmin_string) > 0:
                    cmin = int(cmin_string)
                if len(cmax_string) > 0:
                    cmax = int(cmax_string)
        return token, cmin, cmax

    def _matching_names(self, token):
        """Returns the list of alphabet symbols that match the given
        name (regex, currently can process one '*' wildcard).

        :type token: str
        :param token: The symbol name (pattern) to expand.

        :rtype: list
        :returns: A list of matching names. Empty list if no name matches.
        """
        if not self._has_wildcard(token):
            return [token]

        wildcard_idx = token.index(self.WILDCARD)
        prefix = token[:wildcard_idx]
        if wildcard_idx < len(token) - 1:
            suffix = token[wildcard_idx + 1:]
        else:
            suffix = ''

        # logging.info('DependencyGrammar._matching_names: token {0}, pref={1}, suff={2}'
        #              ''.format(token, prefix, suffix))

        matching_names = list(self.alphabet.keys())
        if len(prefix) > 0:
            matching_names = [n for n in matching_names if n.startswith(prefix)]
        if len(suffix) > 0:
            matching_names = [n for n in matching_names if n.endswith(suffix)]

        return matching_names

    def _validate_rules(self, rules):
        """Check that all the rules are valid under the current alphabet."""
        missing_heads = set()
        missing_children = set()
        for h, ch in rules:
            if h not in self.alphabet:
                missing_heads.add(h)
            if ch not in self.alphabet:
                missing_children.add(ch)

        if (len(missing_heads) + len(missing_children)) > 0:
            logging.warning('DependencyGrammar.validate_rules: missing heads '
                            '{0}, children {1}'
                            ''.format(missing_heads, missing_children))
            return False
        else:
            return True

    def _has_wildcard(self, name):
        return self.WILDCARD in name

    def is_head(self, head, child):
        return (head, child) in self.rules


##############################################################################
# Feature extraction

class PairwiseClfFeatureExtractor(object):
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
        """The call is per item (in this case, CropObject pair)."""
        fd = self.get_features_relative_bbox_and_clsname(*args, **kwargs)
        # Compensate for the vecotrizer "target", which we don't have here (by :-1)
        item_features = self.vectorizer.transform(fd).toarray()[0, :-1]
        return item_features

    def get_features_relative_bbox_and_clsname(self, c_from, c_to):
        """Extract a feature vector from the given pair of CropObjects.
        Does *NOT* convert the class names to integers.

        Features: bbox(c_to) - bbox(c_from), class_name(c_from), class_name(c_to)
        Target: 1 if there is a link from u to v

        Returns a dict that works as input to ``self.vectorizer``.
        """
        target = 0
        if c_from.document == c_to.document:
            if c_to.objid in c_from.outlinks:
                target = 1
        features = (c_to.top - c_from.top,
                    c_to.left - c_from.left,
                    c_to.bottom - c_from.bottom,
                    c_to.right - c_from.right,
                    c_from.clsname,
                    c_to.clsname,
                    target)
        dt, dl, db, dr, cu, cv, tgt = features
        # Normalizing clsnames
        if cu.startswith('letter'): cu = 'letter'
        if cu.startswith('numeral'): cu = 'numeral'
        if cv.startswith('letter'): cv = 'letter'
        if cv.startswith('numeral'): cv = 'numeral'
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
        """Extract a feature vector from the given pair of CropObjects.
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
        if cu.startswith('letter'): cu = 'letter'
        if cu.startswith('numeral'): cu = 'numeral'
        if cv.startswith('letter'): cv = 'letter'
        if cv.startswith('numeral'): cv = 'numeral'
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
    boxes of two CropObjects and their classes and returns whether there
    is an edge or not."""
    MAXIMUM_DISTANCE_THRESHOLD = 200

    def __init__(self, grammar, clf, cropobject_feature_extractor):
        self.grammar = grammar
        self.clf = clf
        self.extractor = cropobject_feature_extractor

    def parse(self, cropobjects):

        # Ensure the same docname for all cropobjects,
        # since we later compute their distances.
        # The correct docname gets set on export anyway.
        default_doc = cropobjects[0].document
        for c in cropobjects:
            c.set_doc(default_doc)

        pairs, features = self.extract_all_pairs(cropobjects)

        logging.info(
            'Clf.Parse: {0} object pairs from {1} objects'.format(len(pairs), len(cropobjects)))

        preds = self.clf.predict(features)

        edges = []
        for idx, (c_from, c_to) in enumerate(pairs):
            if preds[idx] != 0:
                edges.append((c_from.objid, c_to.objid))

        edges = self._apply_trivial_fixes(cropobjects, edges)
        return edges

    def _apply_trivial_fixes(self, cropobjects, edges):
        edges = self._only_one_stem_per_notehead(cropobjects, edges)
        edges = self._every_full_notehead_has_a_stem(cropobjects, edges)

        return edges

    def _only_one_stem_per_notehead(self, nodes: List[Node], edges):
        node_id_to_node_mapping = {n.id: n for n in nodes}  # type: Dict[int, Node]

        # Collect stems per notehead
        stems_per_notehead = collections.defaultdict(list)
        stem_objids = set()
        for from_id, to_id in edges:
            from_node = node_id_to_node_mapping[from_id]
            to_node = node_id_to_node_mapping[to_id]
            if (from_node.class_name in _CONST.NOTEHEAD_CLSNAMES) and \
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

    def _every_full_notehead_has_a_stem(self, nodes: List[Node], edges):
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
                node_id_to_node_mapping[s_objid])
                                     < closest_stem_threshold_distance
                                     }

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

    def is_edge(self, c_from, c_to):
        features = self.extractor(c_from, c_to)
        result = self.clf.predict(features)
        return result

    def set_grammar(self, grammar):
        self.grammar = grammar


def do_parse(nodes: List[Node], parser):
    # names = [c.class_name for c in cropobjects]
    non_staff_cropobjects = [c for c in nodes
                             if c.clsname not in \
                             _CONST.STAFF_CROPOBJECT_CLSNAMES]
    edges = parser.parse(non_staff_cropobjects)
    logging.info('CropObjectListView.parse_selection(): {0} edges to add'
                 ''.format(len(edges)))

    # Add edges
    id_to_node_mapping = {c.objid: c for c in nodes}
    for f, t in edges:
        cf, ct = id_to_node_mapping[f], id_to_node_mapping[t]
        if t not in cf.outlinks:
            if f not in ct.inlinks:
                cf.outlinks.append(t)
                ct.inlinks.append(f)

    return [c for c in list(id_to_node_mapping.values())]


##############################################################################
# Staffline building

def process_stafflines(cropobjects,
                       do_build_staffs=True,
                       do_build_staffspaces=True,
                       do_add_staff_relationships=True):
    """Merges staffline fragments into stafflines. Can group them into staffs,
    add staffspaces, and add the various obligatory relationships of other
    objects to the staff objects. Required before attempting to export MIDI."""
    if len([c for c in cropobjects if c.clsname == 'staff']) > 0:
        logging.warning('Some stafflines have already been processed. Reprocessing'
                     ' is not certain to work.')

    try:
        new_cropobjects = merge_staffline_segments(cropobjects)
    except ValueError as e:
        logging.warning('Model: Staffline merge failed:\n\t\t'
                     '{0}'.format(e.message))
        raise

    try:
        if do_build_staffs:
            staffs = build_staff_nodes(new_cropobjects)
            new_cropobjects = new_cropobjects + staffs
    except Exception as e:
        logging.warning('Building staffline cropobjects from merged segments failed:'
                     ' {0}'.format(e.message))
        raise

    try:
        if do_build_staffspaces:
            staffspaces = build_staffspace_nodes(new_cropobjects)
            new_cropobjects = new_cropobjects + staffspaces
    except Exception as e:
        logging.warning('Building staffspace cropobjects from stafflines failed:'
                     ' {0}'.format(e.message))
        raise

    try:
        if do_add_staff_relationships:
            new_cropobjects = add_staff_relationships(new_cropobjects)
    except Exception as e:
        logging.warning('Adding staff relationships failed:'
                     ' {0}'.format(e.message))
        raise

    return new_cropobjects


def find_wrong_edges(nodes: List[Node], grammar):
    id_to_node_mapping = {node.id: node for node in nodes}
    graph = NotationGraph(nodes)

    incoherent_beam_pairs = find_beams_incoherent_with_stems(nodes)
    # Switching off misdirected leger lines: there is something wrong with them
    misdirected_leger_lines = find_misdirected_leger_line_edges(nodes)

    wrong_edges = [(n.objid, b.objid)
                   for n, b in incoherent_beam_pairs + misdirected_leger_lines]

    disallowed_symbol_class_pairs = [(f, t) for f, t in graph.edges
                                     if not grammar.validate_edge(id_to_node_mapping[f].clsname,
                                                                  id_to_node_mapping[t].clsname)]
    wrong_edges += disallowed_symbol_class_pairs
    return wrong_edges


def find_very_small_cropobjects(cropobjects,
                                bbox_threshold=40, mask_threshold=35):
    very_small_cropobjects = []

    for c in cropobjects:
        total_masked_area = c.mask.sum()
        total_bbox_area = c.width * c.height
        if total_bbox_area < bbox_threshold:
            very_small_cropobjects.append(c)
        elif total_masked_area < mask_threshold:
            very_small_cropobjects.append(c)

    return list(set([c.objid for c in very_small_cropobjects]))


##############################################################################
# Precedence edges

def infer_precedence_edges(nodes: List[Node], factor_by_staff=True):
    """Returns a list of (from_objid, to_objid) parirs. They
    then need to be added to the cropobjects as precedence edges."""
    id_to_node_mapping = {c.id: c for c in nodes}
    _relevant_clsnames = set(list(_CONST.NONGRACE_NOTEHEAD_CLSNAMES)
                             + list(_CONST.REST_CLSNAMES))
    prec_cropobjects = [c for c in nodes
                        if c.clsname in _relevant_clsnames]
    logging.info('_infer_precedence: {0} total prec. cropobjects'
                 ''.format(len(prec_cropobjects)))

    # Group the objects according to the staff they are related to
    # and infer precedence on these subgroups.
    if factor_by_staff:
        staffs = [c for c in nodes
                  if c.clsname == _CONST.STAFF_CLSNAME]
        logging.info('_infer_precedence: got {0} staffs'.format(len(staffs)))
        staff_objids = {c.objid: i for i, c in enumerate(staffs)}
        prec_cropobjects_per_staff = [[] for _ in staffs]
        # All CropObjects relevant for precedence have a relationship
        # to a staff.
        for c in prec_cropobjects:
            for o in c.outlinks:
                if o in staff_objids:
                    prec_cropobjects_per_staff[staff_objids[o]].append(c)

        logging.info('Precedence groups: {0}'
                     ''.format(prec_cropobjects_per_staff))
        prec_edges = []
        for prec_cropobjects_group in prec_cropobjects_per_staff:
            group_prec_edges = infer_precedence_edges(prec_cropobjects_group,
                                                      factor_by_staff=False)
            prec_edges.extend(group_prec_edges)
        return prec_edges

    if len(prec_cropobjects) <= 1:
        logging.info('EdgeListView._infer_precedence: less than 2'
                     ' timed CropObjects selected, no precedence'
                     ' edges to infer.')
        return []

    # Group into equivalence if noteheads share stems
    _stems_to_noteheads_map = collections.defaultdict(list)
    for c in prec_cropobjects:
        for o in c.outlinks:
            if o not in id_to_node_mapping:
                logging.warning('Dangling outlink: {} --> {}'.format(c.objid, o))
                continue
            c_o = id_to_node_mapping[o]
            if c_o.clsname == 'stem':
                _stems_to_noteheads_map[c_o.objid].append(c.objid)

    _prec_equiv_objids = []
    _stemmed_noteheads_objids = []
    for _stem_objid, _stem_notehead_objids in list(_stems_to_noteheads_map.items()):
        _stemmed_noteheads_objids = _stemmed_noteheads_objids \
                                    + _stem_notehead_objids
        _prec_equiv_objids.append(_stem_notehead_objids)
    for c in prec_cropobjects:
        if c.objid not in _stemmed_noteheads_objids:
            _prec_equiv_objids.append([c.objid])

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
                edges.append((f.objid, t.objid))

    return edges


def add_precedence_edges(nodes: List[Node], edges):
    """Adds precedence edges to CropObjects."""
    # Ensure unique
    edges = set(edges)
    id_to_node_mapping = {c.objid: c for c in nodes}

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


##############################################################################
# Build the MIDI


def build_midi(nodes: List[Node], selected_cropobjects=None,
               retain_pitches=True,
               retain_durations=True,
               retain_onsets=True,
               tempo=180):
    """Attempts to export a MIDI file from the current graph. Assumes that
    all the staff objects and their relations have been correctly established,
    and that the correct precedence graph is available.

    :param retain_pitches: If set, will record the pitch information
        in pitched objects.

    :param retain_durations: If set, will record the duration information
        in objects to which it applies.

    :returns: A single-track ``midiutil.MidiFile.MIDIFile`` object. It can be
        written to a stream using its ``mf.writeFile()`` method."""
    id_to_node_mapping = {c.objid: c for c in nodes}

    pitch_inference_engine = PitchInferenceEngine()
    time_inference_engine = OnsetsInferenceEngine(nodes=list(id_to_node_mapping.values()))

    try:
        logging.info('Running pitch inference.')
        pitches, pitch_names = pitch_inference_engine.infer_pitches(list(id_to_node_mapping.values()),
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
        durations = time_inference_engine.durations(list(id_to_node_mapping.values()))
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
        onsets = time_inference_engine.onsets(list(id_to_node_mapping.values()))
    except Exception as e:
        logging.warning('Model: Onset inference failed!')
        logging.exception(traceback.format_exc(e))
        raise

    if retain_onsets:
        for objid in onsets:
            c = id_to_node_mapping[objid]
            c.data['onset_beats'] = onsets[objid]

    # Process ties
    durations, onsets = time_inference_engine.process_ties(list(id_to_node_mapping.values()),
                                                           durations, onsets)

    # Prepare selection subset
    if selected_cropobjects is None:
        selected_cropobjects = list(id_to_node_mapping.values())
    selection_objids = [c.objid for c in selected_cropobjects]

    # Build the MIDI data
    midi_builder = MIDIBuilder()
    mf = midi_builder.build_midi(
        pitches=pitches, durations=durations, onsets=onsets,
        selection=selection_objids, tempo=tempo)

    return mf


##############################################################################


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
    feature_extractor = PairwiseClfFeatureExtractor(vectorizer=vectorizer)

    with open(args.parser) as hdl:
        classifier = pickle.load(hdl)

    mlclass_list = parse_node_classes(args.mlclasses)
    mlclasses = {m.clsid: m for m in mlclass_list}

    grammar = DependencyGrammar(grammar_filename=args.grammar,
                                mlclasses=mlclasses)

    parser = PairwiseClassificationParser(grammar=grammar,
                                          clf=classifier,
                                          cropobject_feature_extractor=feature_extractor)

    #################################################################
    logging.info('Load graph')
    cropobjects = read_nodes_from_file(args.input_mung)

    logging.info('Filter very small')
    very_small_cropobjects = find_very_small_cropobjects(cropobjects,
                                                         bbox_threshold=40,
                                                         mask_threshold=35)
    very_small_cropobjects = set(very_small_cropobjects)
    cropobjects = [c for c in cropobjects if c not in very_small_cropobjects]

    logging.info('Parsing')
    cropobjects = do_parse(cropobjects, parser=parser)

    # Filter contained here.
    if args.filter_contained:
        logging.info('Finding contained cropobjects...')
        contained = find_contained_nodes(cropobjects,
                                         mask_threshold=0.95)
        NEVER_DISCARD_CLASSES = ['key_signature', 'time_signature']
        contained = [c for c in contained if c.clsname not in NEVER_DISCARD_CLASSES]

        _contained_counts = collections.defaultdict(int)
        for c in contained:
            _contained_counts[c.clsname] += 1
        logging.info('Found {} contained cropobjects'.format(len(contained)))
        logging.info('Contained counts:\n{0}'.format(pprint.pformat(dict(_contained_counts))))
        cropobjects = remove_contained_nodes(cropobjects,
                                             contained)
        logging.info('Removed contained cropobjects: {}...'.format([m.objid for m in contained]))

    logging.info('Inferring staffline & staff objects, staff relationships')
    cropobjects = process_stafflines(cropobjects)

    if args.add_key_signatures:
        cropobjects = add_key_signatures(cropobjects)

    logging.info('Filter invalid edges')
    graph = NotationGraph(cropobjects)
    # Operatng on the graph changes the cropobjects
    #  -- the graph only keeps a pointer
    wrong_edges = find_wrong_edges(cropobjects, grammar)
    for f, t in wrong_edges:
        graph.remove_edge(f, t)

    logging.info('Add precedence relationships, factored only by staff')
    prec_edges = infer_precedence_edges(cropobjects)
    cropobjects = add_precedence_edges(cropobjects, prec_edges)

    logging.info('Ensuring MIDI can be built')
    mf = build_midi(cropobjects,
                    retain_pitches=True,
                    retain_durations=True,
                    retain_onsets=True,
                    tempo=180)

    logging.info('Save output')
    docname = os.path.splitext(os.path.basename(args.output_mung))[0]
    xml = export_node_list(cropobjects,
                           document=docname,
                           dataset='FNOMR_results')
    with open(args.output_mung, 'wb') as out_stream:
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
