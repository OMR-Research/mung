"""This module implements a Grammar.

A Grammar is a set of rules about how objects from a certain set
of classes are allowed to form relationships. In a dependency grammar,
the relationships are formed directly between the objects. (In
constituency grammars, we'd have a "merge result" object instead.)

In the ``mung`` package, you can use grammars to validate
whether the relationships between the annotated objects conform
to the specification.

.. warning::

    The grammar is not a formal specification. Music notation sometimes
    breaks its own rules. More importantly, people who write music
    notation by hand make mistakes. This means that not all annotation
    files will pass grammar validation without errors, and that is fine.
    If this bothers you, use the MUSCIMarker tool to visualize the errors.


.. TODO::
    create image:: ../doc/_static/grammar_explainer.png

"""
import codecs
import logging
import os
import pprint

import collections

from typing import Tuple, List, Any, Set, Dict


class DependencyGrammarParseError(ValueError):
    pass


class DependencyGrammar(object):
    """The DependencyGrammar class implements rules about valid graphs above
    objects from a set of recognized classes.

    The Grammar complements a Parser. It defines rules, and the Parser
    implements algorithms to apply these rules to some input.

    A grammar has an **Alphabet** and **Rules**. The alphabet is a list
    of symbols that the grammar recognizes. Rules are constraints on
    the structures that can be induced among these symbols.

    There are two kinds of grammars according to what kinds of rules
    they use: **dependency** rules, and **constituency** rules. We use
    dependency grammars. Dependency grammar rules specify which symbols
    are governing, and which symbols are governed::

      noteheadFull | stem

    There can be multiple left-hand side and right-hand side symbols,
    as a shortcut for a list of rules::

        noteheadFull | stem beam
        noteheadFull noteheadHalf | legerLine durationDot tie notehead*Small

    The asterisk works as a wildcard. Currently, only one wildcard per symbol
    is allowed::

      timeSignature | numeral*

    Lines starting with a ``#`` are regarded as comments and ignored.
    Empty lines are also ignored.

    **Cardinality rules**

    We can also specify in the grammar the minimum and/or maximum number
    of relationships, both inlinks and outlinks, that an object can form
    with other objects of given types. For example:

    * One notehead may have up to two stems attached.
    * We also allow for stemless full noteheads.
    * One stem can be attached to multiple noteheads, but at least one.

    This would be expressed as::

      notehead*{,2} | stem{1,}

    The relationship of noteheads to leger lines is generally ``m:n``::

      noteheadFull | legerLine

    A time signature may consist of multiple numerals, but only one
    other symbol::

      timeSignature{1,} | numeral*{1}
      timeSignature{1} | timeSigCommon timeSigCutCommon

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

    And analogously for inlinks::

      | letter*{1,}
      | numeral*{1,}
      | legerLine{1,}
      | noteheadFullSmall{1,}

    **Interface**

    The basic role of the dependency grammar is to provide the list of rules:

    >>> from mung.io import parse_node_classes
    >>> filepath = os.path.dirname(os.path.dirname(__file__)) + u'/test/test_data/mff-muscima-classes-annot.deprules'
    >>> node_classes_path = os.path.dirname(os.path.dirname(__file__)) + u'/test/test_data/mff-muscima-classes-annot.xml'
    >>> node_classes = parse_node_classes(node_classes_path)
    >>> node_classes_dict = {node_class.name for node_class in node_classes}
    >>> dependency_graph = DependencyGrammar(grammar_filename=filepath, alphabet=node_classes_dict)
    >>> len(dependency_graph.rules)
    646

    The grammar can validate against these rules. The workhorse of this
    functionality is the ``find_invalid_in_graph()`` method, which finds
    objects that have inlinks/outlinks which do not comply with the grammar,
    and the non-compliant inlinks/outlinks as well.

    If we have the following notation objects ``0``, ``1``, ``2``, and ``3``,
    with the following symbol classes:

    >>> vertices = {0: 'noteheadFull', 1: 'stem', 2: 'flag8thUp', 3: 'noteheadHalf'}

    And the following relationships were recorded:

    >>> edges = [(0, 1), (0, 2), (0, 3)]

    We can check for errors against our music notation symbols dependency
    grammar:

    >>> wrong_vertices, wrong_inlinks, wrong_outlinks, _, _, _ = \
            dependency_graph.find_invalid_in_graph(vertices=vertices, edges=edges)

    Because the edge ``(0, 3)`` connects a full notehead to an empty notehead,
    the method should report the objects ``0`` and ``3`` as wrong, as well
    as the corresponding inlink of ``3`` and outlink of ``0``:

    >>> wrong_vertices
    [0, 3]
    >>> wrong_inlinks
    [(0, 3)]
    >>> wrong_outlinks
    [(0, 3)]

    (Note that both the inlinks and outlinks are recorded in a ``(from, to)``
    format.)

    .. caution::

        Aside from checking against illegal relationships (such as we
        saw in the example), errors can also come from too many or too
        few inlinks/outlinks of a given type. However,
        the validation currently implements checks only for aggregate
        cardinalities, not for pair cardinalities (so, there can be
        e.g. multiple sharps attached to a notehead, even though the cardinality
        in the ``notehead | sharp`` rule is set to max. 1).

    **Grammar file formats**

    The alphabet is stored by means of a ``NodeClassList`` XML file with
    :class:`NodeClass` elements, as described in the :mod:`mung.io` module.

    The rules are stored in *rule files*, with the suffix ``.deprules``.

    A rule file line can be empty, start with a ``#`` (comment), or contain
    a rule symbol ``|``. Empty lines and comments are ignored during parsing.
    Rules are split into left- and right-hand side tokens, according to
    the position of the ``|`` symbol.

    Parsing a token returns the token string (unexpanded wildcards), its
    minimum and maximum cardinality in the rule (defaults are ``(0, 10000)``
    if no cardinality is provided).

    >>> dependency_graph.parse_token('notehead*')
    ('notehead*', 0, 10000)
    >>> dependency_graph.parse_token('notehead*{1,5}')
    ('notehead*', 1, 5)
    >>> dependency_graph.parse_token('notehead*{1,}')
    ('notehead*', 1, 10000)
    >>> dependency_graph.parse_token('notehead*{,5}')
    ('notehead*', 0, 5)
    >>> dependency_graph.parse_token('notehead*{1}')
    ('notehead*', 1, 1)

    The wildcards are expanded at the level of a line.

    >>> l = 'notehead*{,2} | stem'
    >>> rules, inlink_cards, outlink_cards, _, _ = dependency_graph.parse_dependency_grammar_line(l)
    >>> rules
    [('noteheadFull', 'stem'), ('noteheadFullSmall', 'stem'), ('noteheadHalf', 'stem'), ('noteheadHalfSmall', 'stem'), ('noteheadWhole', 'stem')]
    >>> outlink_cards['noteheadHalf'] == {'stem': (0, 2)}
    True
    >>> inlink_cards['stem'] == {'noteheadHalf': (0, 10000), 'noteheadFull': (0, 10000), 'noteheadWhole': (0, 10000), 'noteheadFullSmall': (0, 10000), 'noteheadHalfSmall': (0, 10000)}
    True


    A key signature can have any number of sharps, flats, or naturals,
    but if a given symbol is part of a key signature, it can only be part of one.

    >>> l = 'keySignature | accidentalSharp{1} accidentalFlat{1} accidentalNatural{1}'
    >>> rules, inlink_cards, _, _, _ = dependency_graph.parse_dependency_grammar_line(l)
    >>> rules
    [('keySignature', 'accidentalFlat'), ('keySignature', 'accidentalNatural'), ('keySignature', 'accidentalSharp')]
    >>> inlink_cards == {'accidentalNatural': {'keySignature': (1, 1)},
    ...                  'accidentalSharp': {'keySignature': (1, 1)},
    ...                  'accidentalFlat': {'keySignature': (1, 1)}}
    True


    You can also give *aggregate* cardinality rules, of the style "whatever rule
    applies, there should be at least X/at most Y edges for this type of object".
    (If no maximum is specified, the value of ``DependencyGrammar._MAX_CARD``
    is used, which is by default 10000).

    >>> l = 'keySignature{1,} |'
    >>> _, _, _, _, out_aggregate_cards = dependency_graph.parse_dependency_grammar_line(l)
    >>> out_aggregate_cards == {'keySignature': (1, 10000)}
    True
    >>> l = 'notehead*Small{1,} |'
    >>> _, _, _, _, out_aggregate_cards = dependency_graph.parse_dependency_grammar_line(l)
    >>> out_aggregate_cards == {'noteheadFullSmall': (1, 10000), 'noteheadHalfSmall': (1, 10000)}
    True
    >>> l = '| beam{1,} stem{1,} accidentalFlat{1,}'
    >>> _, _, _, in_aggregate_cards, _ = dependency_graph.parse_dependency_grammar_line(l)
    >>> in_aggregate_cards == {'stem': (1, 10000), 'beam': (1, 10000), 'accidentalFlat': (1, 10000)}
    True

    """

    WILDCARD = '*'

    _MAX_CARDINALITY = 10000

    def __init__(self, grammar_filename: str, alphabet: Set[str]):
        """Initialize the Grammar: fill in alphabet and parse rules.

        :param grammar_filename: Path to a file that contains deprules
            (see class documentation for ``*.deprules`` file format).

        :param alphabet: A set or list of symbol class names, which
            are used in the *.deprules file.
        """
        self.alphabet = set(alphabet)
        # logging.info('DependencyGrammar: got alphabet:\n{0}'
        #              ''.format(pprint.pformat(self.alphabet)))
        self.rules = []  # type: List[Tuple[str, str]]
        self.inlink_cardinalities = {}  # type: Dict[str, Dict[str, Tuple[int, int]]]
        self.outlink_cardinalities = {}  # type: Dict[str, Dict[str, Tuple[int, int]]]
        self.inlink_aggregated_cardinalities = {}  # type: Dict[str, Tuple[int, int]]
        self.outlink_aggregated_cardinalities = {}  # type: Dict[str, Tuple[int, int]]

        rules, inlink_cardinalities, outlink_cardinalities, inlink_aggregated_cardinalitites, \
        outlink_aggregated_cardinalitites = self.parse_dependency_grammar_rules(grammar_filename)

        if self.__validate_rules(rules):
            self.rules = rules
            logging.info('DependencyGrammar: Imported {0} rules'
                         ''.format(len(self.rules)))
            self.inlink_cardinalities = inlink_cardinalities
            self.outlink_cardinalities = outlink_cardinalities
            self.inlink_aggregated_cardinalities = inlink_aggregated_cardinalitites
            self.outlink_aggregated_cardinalities = outlink_aggregated_cardinalitites
            logging.debug('DependencyGrammar: Inlink aggregated cardinalities: {0}'
                          ''.format(pprint.pformat(inlink_aggregated_cardinalitites)))
            logging.debug('DependencyGrammar: Outlink aggregated cardinalities: {0}'
                          ''.format(pprint.pformat(outlink_aggregated_cardinalitites)))
        else:
            raise DependencyGrammarParseError(
                'Not able to parse dependency grammar file {0}.'
                ''.format(grammar_filename))

    def validate_edge(self, head_name: str, child_name: str) -> bool:
        """Check whether a given ``head --> child`` edge conforms
        with this grammar."""
        return (head_name, child_name) in self.rules

    def validate_graph(self, vertices: Dict[int, str], edges: List[Tuple[int, int]]):
        """Checks whether the given graph complies with the grammar.

        :param vertices: A dict with any keys and values corresponding
            to the alphabet of the grammar.

        :param edges: A list of ``(from, to)`` pairs, where both
            ``from`` and ``to`` are valid keys into the ``vertices`` dict.

        :returns: ``True`` if the graph is valid, ``False`` otherwise.
        """
        v, i, o, _, _, _ = self.find_invalid_in_graph(vertices=vertices, edges=edges)
        return len(v) == 0

    def find_invalid_in_graph(self, vertices: Dict[int, str], edges: List[Tuple[int, int]]) \
            -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]], Dict[int, str], Dict[
                Tuple[int, int], str], Dict[Tuple[int, int], str]]:
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

        :param vertices: A dict with any keys, and values corresponding
            to the alphabet of the grammar.

        :param edges: A list of ``(from, to)`` pairs, where both
            ``from`` and ``to`` are valid keys into the ``vertices`` dict.

        :param provide_reasons: If set, will generate string descriptions
            of each error and return them.

        :returns: A list of vertices, a list of inlinks and a list of outlinks
            that do not comply with the grammar as well as three dictionaries with reasons
            for each list respectively.
        """
        logging.info('DependencyGrammar: looking for errors.')

        wrong_vertices = []  # type: List[int]
        wrong_inlinks = []  # type: List[Tuple[int,int]]
        wrong_outlinks = []  # type: List[Tuple[int,int]]

        reasons_incorrect_vertices = {}  # type: Dict[int, str]
        reasons_incorrect_inlinks = {}  # type: Dict[Tuple[int,int], str]
        reasons_incorrect_outlinks = {}  # type: Dict[Tuple[int,int], str]

        # Check that vertices have labels that are in the alphabet
        for v, class_name in list(vertices.items()):
            if class_name not in self.alphabet:
                wrong_vertices.append(v)
                reasons_incorrect_vertices[v] = 'Symbol {0} not in alphabet: class {1}.' \
                                                ''.format(v, class_name)

        # Check that all edges are allowed
        for from_id, to_id in edges:
            from_class_name, to_class_name = str(vertices[from_id]), str(vertices[to_id])
            if (from_class_name, to_class_name) not in self.rules:
                logging.debug('Wrong edge: {0} --> {1}, rules:\n{2}'
                              ''.format(from_class_name, to_class_name, pprint.pformat(self.rules)))

                wrong_inlinks.append((from_id, to_id))
                reasons_incorrect_inlinks[(from_id, to_id)] = 'Outlink {0} ({1}) -> {2} ({3}) not in alphabet.'.format(
                    from_class_name, from_id, to_class_name, to_id)

                wrong_outlinks.append((from_id, to_id))
                reasons_incorrect_outlinks[(from_id, to_id)] = 'Outlink {0} ({1}) -> {2} ({3}) not in alphabet.'.format(
                    from_class_name, from_id, to_class_name, to_id)
                if from_id not in wrong_vertices:
                    wrong_vertices.append(from_id)
                    reasons_incorrect_vertices[from_id] = 'Symbol {0} (class: {1}) participates ' \
                                                          'in wrong outlink: {2} ({3}) --> {4} ({5})' \
                                                          ''.format(from_id, vertices[from_id],
                                                                    from_class_name, from_id,
                                                                    to_class_name, to_id)
                if to_id not in wrong_vertices:
                    wrong_vertices.append(to_id)
                    reasons_incorrect_vertices[to_id] = 'Symbol {0} (class: {1}) participates ' \
                                                        'in wrong inlink: {2} ({3}) --> {4} ({5})' \
                                                        ''.format(to_id, vertices[to_id],
                                                                  from_class_name, from_id,
                                                                  to_class_name, to_id)

        # Check aggregate cardinality rules
        #  - build inlink and outlink dicts
        inlinks = {}
        outlinks = {}
        for v in vertices:
            outlinks[v] = set()
            inlinks[v] = set()
        for from_id, to_id in edges:
            outlinks[from_id].add(to_id)
            inlinks[to_id].add(from_id)

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
        logging.info('DependencyGrammar: checking outlink aggregate cardinalities'
                     '\n{0}'.format(pprint.pformat(outlinks)))
        for from_id in outlinks:
            from_class_name = vertices[from_id]
            if from_class_name not in self.outlink_aggregated_cardinalities:
                # Given vertex has no aggregate cardinality restrictions
                continue
            cmin, cmax = self.outlink_aggregated_cardinalities[from_class_name]
            logging.info('DependencyGrammar: checking outlink cardinality'
                         ' rule fulfilled for vertex {0} ({1}): should be'
                         ' within {2} -- {3}'.format(from_id, vertices[from_id], cmin, cmax))
            if not (cmin <= len(outlinks[from_id]) <= cmax):
                wrong_vertices.append(from_id)
                reasons_incorrect_vertices[from_id] = 'Symbol {0} (class: {1}) has {2} outlinks,' \
                                                      ' but grammar specifies {3} -- {4}.' \
                                                      ''.format(from_id, vertices[from_id],
                                                                len(outlinks[from_id]),
                                                                cmin, cmax)

        for to_id in inlinks:
            to_class_name = vertices[to_id]
            if to_class_name not in self.inlink_aggregated_cardinalities:
                continue
            cmin, cmax = self.inlink_aggregated_cardinalities[to_class_name]
            if not (cmin <= len(inlinks[to_id]) <= cmax):
                wrong_vertices.append(to_id)
                reasons_incorrect_vertices[to_id] = 'Symbol {0} (class: {1}) has {2} inlinks,' \
                                                    ' but grammar specifies {3} -- {4}.' \
                                                    ''.format(to_id, vertices[to_id],
                                                              len(inlinks[to_id]),
                                                              cmin, cmax)

        return wrong_vertices, wrong_inlinks, wrong_outlinks, reasons_incorrect_vertices, reasons_incorrect_inlinks, reasons_incorrect_outlinks

    def parse_dependency_grammar_rules(self, filename: str) -> \
            Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Tuple[int, int]]], Dict[str, Dict[str, Tuple[int, int]]],
                  Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
        """Returns the rules stored in the given rule file.

        A dependency grammar rule file contains grammar lines,
        comment lines, and other lines. A grammar line is any line that
        contains the ``|`` symbol and does *not* have a ``#`` as the first
        non-whitespace symbol.

        Comment lines are those that have ``#`` as the first non-whitespace
        symbol. They are ignored, even if they contain ``|``. All other lines
        that do contain ``|`` are considered to be grammar lines.

        All lines that do not contain ``|`` are considered non-grammar lines
        and are ignored.

        :param filename: The path to the rule file.

        :returns: A quintuplet of the grammar rules.

            * ``rules``: a list of ``(from_class, to_class)`` tuples. Each rule tuple
              encodes that relationships leading from symbols of type ``from_class``
              to symbols of type ``to_class`` may exist.
            * ``inlink_cards``: a dictionary that encodes the range of permitted cardinalities
              for each RHS symbol of *inlinks* from the LHS symbols.
            * ``outlink_cards``: a dictionary that encodes the range of permitted cardinalities
              for each LHS of *outlinks* to the RHS symbols.
            * ``inlink_aggregate_cards``: A dict that holds for each RHS the range of
              permitted total inlink counts. E.g., a stem must always have at least one inlink.
            * ``outlink_aggregate_cards``: A dict that holds for each LHS the range
              of permitted total outlink counts. E.g., a full notehead must always have
              at least one outlink.

        """
        rules = []
        inlink_cardinalities = collections.OrderedDict()
        outlink_cardinalities = collections.OrderedDict()

        inlink_aggregated_cardinalities = collections.OrderedDict()
        outlink_aggregated_cardinalities = collections.OrderedDict()

        _invalid_lines = []
        with codecs.open(filename, 'r', 'utf-8') as hdl:
            for line_no, line in enumerate(hdl):
                l_rules, in_card, out_card, in_agg_card, out_agg_card = self.parse_dependency_grammar_line(
                    line)

                if not self.__validate_rules(l_rules):
                    _invalid_lines.append((line_no, line))

                rules.extend(l_rules)

                # Update cardinalities
                for lhs in out_card:
                    if lhs not in outlink_cardinalities:
                        outlink_cardinalities[lhs] = collections.OrderedDict()
                    outlink_cardinalities[lhs].update(out_card[lhs])

                for rhs in in_card:
                    if rhs not in inlink_cardinalities:
                        inlink_cardinalities[rhs] = collections.OrderedDict()
                    inlink_cardinalities[rhs].update(in_card[rhs])

                inlink_aggregated_cardinalities.update(in_agg_card)
                outlink_aggregated_cardinalities.update(out_agg_card)

        if len(_invalid_lines) > 0:
            logging.warning('DependencyGrammar.parse_rules(): Invalid lines'
                            ' {0}'.format(pprint.pformat(_invalid_lines)))

        return rules, inlink_cardinalities, outlink_cardinalities, inlink_aggregated_cardinalities, outlink_aggregated_cardinalities

    def parse_dependency_grammar_line(self, line: str) -> \
            Tuple[List[Tuple[str, str]], Dict[str, Dict[str, Tuple[int, int]]], Dict[
                str, Dict[str, Tuple[int, int]]], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
        """Parse one dependency grammar line. See DependencyGrammar
        I/O documentation for the full format description of valid
        grammar lines.

        The grammar line specifies two kinds of information: which symbol
        classes may form relationships, and what the valid cardinalities
        for these relationships are. For instance, while ``time_signature``
        symbols have outlinks to ``numeral_X`` symbols, one numeral cannot
        be part of more than one time signature.

        A grammar line has a left-hand side (lhs) and a right-hand side (rhs),
        separated by the ``|`` symbol.

        (See :class:`DependencyGramamr` documentation for examples.)

        :param line: One line of a dependency grammar rule file.

        :returns: A quintuplet of:

            * ``rules``: a list of ``(from_class, to_class)`` tuples. Each rule tuple
              encodes that relationships leading from symbols of type ``from_class``
              to symbols of type ``to_class`` may exist.
            * ``inlink_cards``: a dictionary that encodes the range of permitted cardinalities
              for each RHS symbol of *inlinks* from the LHS symbols.
            * ``outlink_cards``: a dictionary that encodes the range of permitted cardinalities
              for each LHS of *outlinks* to the RHS symbols.
            * ``inlink_aggregate_cards``: A dict that holds for each RHS the range of
              permitted total inlink counts. E.g., a stem must always have at least one inlink.
            * ``outlink_aggregate_cards``: A dict that holds for each LHS the range
              of permitted total outlink counts. E.g., a full notehead must always have
              at least one outlink.

            For non-grammar lines (see :meth:`parse_dependency_grammar_rules`),
            this method returns empty data structures.

        """
        rules = []
        out_cards = collections.OrderedDict()
        in_cards = collections.OrderedDict()
        out_agg_cards = collections.OrderedDict()
        in_agg_cards = collections.OrderedDict()

        _no_rule_line_output = [], collections.OrderedDict(), collections.OrderedDict(), \
                               collections.OrderedDict(), collections.OrderedDict()
        if line.strip().startswith('#'):
            return _no_rule_line_output
        if len(line.strip()) == 0:
            return _no_rule_line_output
        if '|' not in line:
            return _no_rule_line_output

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
                for t in self.__matching_names(token):
                    in_agg_cards[t] = (rhs_cmin, rhs_cmax)
            logging.debug('DependencyGrammar: found inlinks: {0}'
                          ''.format(pprint.pformat(in_agg_cards)))
            return rules, out_cards, in_cards, in_agg_cards, out_agg_cards

        if _line_type == 'aggregate_outlinks':
            lhs_tokens = lhs.strip().split()
            for lt in lhs_tokens:
                token, lhs_cmin, lhs_cmax = self.parse_token(lt.strip())
                for t in self.__matching_names(token):
                    out_agg_cards[t] = (lhs_cmin, lhs_cmax)
            logging.debug('DependencyGrammar: found outlinks: {0}'
                          ''.format(pprint.pformat(out_agg_cards)))
            return rules, out_cards, in_cards, in_agg_cards, out_agg_cards

        # Normal line that defines a left-hand side and a right-hand side

        lhs_symbols = []
        # These cardinalities apply to all left-hand side tokens,
        # for edges leading to any of the right-hand side tokens.
        lhs_cards = collections.OrderedDict()
        for l in lhs_tokens:
            token, lhs_cmin, lhs_cmax = self.parse_token(l)
            all_tokens = self.__matching_names(token)
            lhs_symbols.extend(all_tokens)
            for t in all_tokens:
                lhs_cards[t] = (lhs_cmin, lhs_cmax)

        rhs_symbols = []
        rhs_cards = collections.OrderedDict()
        for r in rhs_tokens:
            token, rhs_cmin, rhs_cmax = self.parse_token(r)
            all_tokens = self.__matching_names(token)
            rhs_symbols.extend(all_tokens)
            for t in all_tokens:
                rhs_cards[t] = (rhs_cmin, rhs_cmax)

        # Build the outputs from the cartesian product
        # of left-hand and right-hand tokens.
        for l in lhs_symbols:
            if l not in out_cards:
                out_cards[l] = collections.OrderedDict()
            for r in rhs_symbols:
                if r not in in_cards:
                    in_cards[r] = collections.OrderedDict()

                rules.append((l, r))
                out_cards[l][r] = lhs_cards[l]
                in_cards[r][l] = rhs_cards[r]

        # Fixed rule ordering
        rules = sorted(rules)

        return rules, in_cards, out_cards, in_agg_cards, out_agg_cards

    def parse_token(self, token: str):
        """Parse one ``*.deprules`` file token. See class documentation for
        examples.

        :param token: One token of a ``*.deprules`` file.

        :return: token, cmin, cmax
        """
        token = str(token)
        cmin, cmax = 0, self._MAX_CARDINALITY
        if '{' not in token:
            token = token
        else:
            token, cardinality = token[:-1].split('{')
            if ',' not in cardinality:
                cmin, cmax = int(cardinality), int(cardinality)
            else:
                cmin_string, cmax_string = cardinality.split(',')
                if len(cmin_string) > 0:
                    cmin = int(cmin_string)
                if len(cmax_string) > 0:
                    cmax = int(cmax_string)
        return token, cmin, cmax

    def __matching_names(self, token):
        """Returns the list of alphabet symbols that match the given
        name (regex, currently can process one '*' wildcard).

        :type token: str
        :param token: The symbol name (pattern) to expand.

        :rtype: list
        :returns: A list of matching names. Empty list if no name matches.
        """
        if not self.__has_wildcard(token):
            return [token]

        wildcard_idx = token.index(self.WILDCARD)
        prefix = token[:wildcard_idx]
        if wildcard_idx < len(token) - 1:
            suffix = token[wildcard_idx + 1:]
        else:
            suffix = ''

        matching_names = list(self.alphabet)
        if len(prefix) > 0:
            matching_names = [n for n in matching_names if n.startswith(prefix)]
        if len(suffix) > 0:
            matching_names = [n for n in matching_names if n.endswith(suffix)]

        return matching_names

    def __validate_rules(self, rules: List[Tuple[str, str]]) -> bool:
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

    def __has_wildcard(self, name):
        return self.WILDCARD in name

    def is_head(self, head, child):
        return (head, child) in self.rules
