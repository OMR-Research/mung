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

      notehead_full | stem

    There can be multiple left-hand side and right-hand side symbols,
    as a shortcut for a list of rules::

        notehead_full | stem beam
        notehead_full notehead_empty | ledger_line duration-dot tie grace_note

    The asterisk works as a wildcard. Currently, only one wildcard per symbol
    is allowed::

      time_signature | numeral_*

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

      notehead-*{,2} | stem{1,}

    The relationship of noteheads to ledger lines is generally ``m:n``::

      noteheadFull | ledger_line

    A time signature may consist of multiple numerals, but only one
    other symbol::

      time_signature{1,} | numeral_*{1}
      time_signature{1} | whole-time_mark alla_breve other_time_signature

    A key signature may have any number of sharps and flats.
    A sharp or flat can only belong to one key signature. However,
    not every sharp belongs to a key signature::

      key_signature | sharp{,1} flat{,1} natural{,1} double_sharp{,1} double_flat{,1}

    For the left-hand side of the rule, the cardinality restrictions apply to
    outlinks towards symbols of classes on the right-hand side of the rule.
    For the right-hand side, the cardinality restrictions apply to inlinks
    from symbols of left-hand side classes.

    It is also possible to specify that regardless of where outlinks
    lead, a symbol should always have at least some::

      time_signature{1,} |
      repeat{2,} |

    And analogously for inlinks::

      | letter_*{1,}
      | numeral_*{1,}
      | ledger_line{1,}
      | grace-notehead-*{1,}

    **Interface**

    The basic role of the dependency grammar is to provide the list of rules:

    >>> from mung.io import parse_node_classes
    >>> fpath = os.path.dirname(os.path.dirname(__file__)) + u'/test/test_data/mff-muscima-classes-annot.deprules'
    >>> node_classes_path = os.path.dirname(os.path.dirname(__file__)) + u'/test/test_data/mff-muscima-classes-annot.xml'
    >>> node_classes_dict = {m.name for m in parse_node_classes(node_classes_path)}
    >>> g = DependencyGrammar(grammar_filename=fpath, alphabet=node_classes_dict)
    >>> len(g.rules)
    551

    The grammar can validate against these rules. The workhorse of this
    functionality is the ``find_invalid_in_graph()`` method, which finds
    objects that have inlinks/outlinks which do not comply with the grammar,
    and the non-compliant inlinks/outlinks as well.

    If we have the following notation objects ``0``, ``1``, ``2``, and ``3``,
    with the following symbol classes:

    >>> vertices = {0: 'noteheadFull', 1: 'stem', 2: '8th_flag', 3: 'notehead_empty'}

    And the following relationships were recorded:

    >>> edges = [(0, 1), (0, 2), (0, 3)]

    We can check for errors against our music notation symbols dependency
    grammar:

    >>> wrong_vertices, wrong_inlinks, wrong_outlinks = \
            g.find_invalid_in_graph(vertices=vertices, edges=edges)

    Because the edge ``(0, 3)`` connects a full notehead to an empty notehead,
    the method should report the objects ``0`` and ``3`` as wrong, as well
    as the corresponding inlink of ``3`` and outlink of ``0``:

    >>> wrong_vertices
    [3, 0, 1]
    >>> wrong_inlinks
    [(0, 1), (0, 3)]
    >>> wrong_outlinks
    [(0, 1), (0, 3)]

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

    The alphabet is stored by means of a ``CropObjectClassList`` XML file with
    :class:`NodeClass` elements, as described in the :mod:`mung.io` module.

    The rules are stored in *rule files*, with the suffix ``.deprules``.

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
    [('notehead-empty', 'stem'), ('noteheadFull', 'stem')]
    >>> outlink_cards['notehead-empty'] == {'stem': (0, 2)}
    True
    >>> inlink_cards['stem'] == {'notehead-empty': (0, 10000), 'noteheadFull': (0, 10000)}
    True


    A key signature can have any number of sharps, flats, or naturals,
    but if a given symbol is part of a key signature, it can only be part of one.

    >>> l = 'key-signature | sharp{1} flat{1} natural{1}'
    >>> rules, inlink_cards, _, _, _ = g.parse_dependency_grammar_line(l)
    >>> rules
    [('key-signature', 'flat'), ('key-signature', 'natural'), ('key-signature', 'sharp')]
    >>> inlink_cards == {'natural': {'key-signature': (1, 1)},
    ...                  'sharp': {'key-signature': (1, 1)},
    ...                  'flat': {'key-signature': (1, 1)}}
    True


    You can also give *aggregate* cardinality rules, of the style "whatever rule
    applies, there should be at least X/at most Y edges for this type of object".
    (If no maximum is specified, the value of ``DependencyGrammar._MAX_CARD``
    is used, which is by default 10000).

    >>> l = 'key-signature{1,} |'
    >>> _, _, _, _, out_aggregate_cards = g.parse_dependency_grammar_line(l)
    >>> out_aggregate_cards == {'key-signature': (1, 10000)}
    True
    >>> l = 'grace-notehead*{1,} |'
    >>> _, _, _, _, out_aggregate_cards = g.parse_dependency_grammar_line(l)
    >>> out_aggregate_cards == {'grace-notehead-empty': (1, 10000), 'grace-notehead-full': (1, 10000)}
    True
    >>> l = '| beam{1,} stem{1,} flat{1,}'
    >>> _, _, _, in_aggregate_cards, _ = g.parse_dependency_grammar_line(l)
    >>> in_aggregate_cards == {'stem': (1, 10000), 'beam': (1, 10000), 'flat': (1, 10000)}
    True

    """

    WILDCARD = '*'

    _MAX_CARD = 10000

    def __init__(self, grammar_filename, alphabet):
        """Initialize the Grammar: fill in alphabet and parse rules.

        :param grammar_filename: Path to a file that contains deprules
            (see class documentation for ``*.deprules`` file format).

        :param alphabet: A set or list of symbol class names, which
            are used in the *.deprules file.
        """
        self.alphabet = set(alphabet)
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
            raise DependencyGrammarParseError(
                'Not able to parse dependency grammar file {0}.'
                ''.format(grammar_filename))

    def validate_edge(self, head_name, child_name):
        """Check whether a given ``head --> child`` edge conforms
        with this grammar."""
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

        :param vertices: A dict with any keys, and values corresponding
            to the alphabet of the grammar.

        :param edges: A list of ``(from, to)`` pairs, where both
            ``from`` and ``to`` are valid keys into the ``vertices`` dict.

        :param provide_reasons: If set, will generate string descriptions
            of each error and return them.

        :returns: A list of vertices, a list of inlinks and a list of outlinks
            that do not comply with the grammar. If ``provide_reasons`` is set,
            also returns three more: dicts of written reasons for each error
            (vertex, inlink, outlink).
        """
        logging.info('DependencyGrammar: looking for errors.')

        wrong_vertices = []
        wrong_inlinks = []
        wrong_outlinks = []

        reasons_v = {}
        reasons_i = {}
        reasons_o = {}

        # Check that vertices have labels that are in the alphabet
        for v, clsname in list(vertices.items()):
            if clsname not in self.alphabet:
                wrong_vertices.append(v)
                reasons_v[v] = 'Symbol {0} not in alphabet: class {1}.' \
                               ''.format(v, clsname)

        # Check that all edges are allowed
        for f, t in edges:
            nf, nt = str(vertices[f]), str(vertices[t])
            if (nf, nt) not in self.rules:
                logging.debug('Wrong edge: {0} --> {1}, rules:\n{2}'
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
        logging.info('DependencyGrammar: checking outlink aggregate cardinalities'
                     '\n{0}'.format(pprint.pformat(outlinks)))
        for f in outlinks:
            f_clsname = vertices[f]
            if f_clsname not in self.outlink_aggregated_cardinalities:
                # Given vertex has no aggregate cardinality restrictions
                continue
            cmin, cmax = self.outlink_aggregated_cardinalities[f_clsname]
            logging.info('DependencyGrammar: checking outlink cardinality'
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
        #for f in outlinks:
        #    oc = self.outlink_cardinalities[f]
        if provide_reasons:
            return wrong_vertices, wrong_inlinks, wrong_outlinks, \
                   reasons_v, reasons_i, reasons_o

        return wrong_vertices, wrong_inlinks, wrong_outlinks

    def parse_dependency_grammar_rules(self, filename):
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
                l_rules, in_card, out_card, in_agg_card, out_agg_card = self.parse_dependency_grammar_line(line)

                if not self._validate_rules(l_rules):
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

        return rules, inlink_cardinalities, outlink_cardinalities, \
               inlink_aggregated_cardinalities, outlink_aggregated_cardinalities

    def parse_dependency_grammar_line(self, line):
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
        :type line: str

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

        _no_rule_line_output = [], collections.OrderedDict(), collections.OrderedDict(),\
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

        #logging.info('DependencyGrammar: tokens lhs={0}, rhs={1}'
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
        lhs_cards = collections.OrderedDict()
        for l in lhs_tokens:
            token, lhs_cmin, lhs_cmax = self.parse_token(l)
            all_tokens = self._matching_names(token)
            lhs_symbols.extend(all_tokens)
            for t in all_tokens:
                lhs_cards[t] = (lhs_cmin, lhs_cmax)

        rhs_symbols = []
        rhs_cards = collections.OrderedDict()
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
                out_cards[l] = collections.OrderedDict()
            for r in rhs_symbols:
                if r not in in_cards:
                    in_cards[r] = collections.OrderedDict()

                rules.append((l, r))
                out_cards[l][r] = lhs_cards[l]
                in_cards[r][l] = rhs_cards[r]

        # logging.info('DependencyGramamr: got rules:\n{0}'
        #              ''.format(pprint.pformat(rules)))
        # logging.info('DependencyGrammar: got inlink cardinalities:\n{0}'
        #              ''.format(pprint.pformat(in_cards)))
        # logging.info('DependencyGrammar: got outlink cardinalities:\n{0}'
        #              ''.format(pprint.pformat(out_cards)))

        # Fixed rule ordering
        rules = sorted(rules)

        return rules, in_cards, out_cards, in_agg_cards, out_agg_cards

    def parse_token(self, l):
        """Parse one ``*.deprules`` file token. See class documentation for
        examples.

        :param l: One token of a ``*.deprules`` file.

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
            suffix = token[wildcard_idx+1:]
        else:
            suffix = ''

        # logging.info('DependencyGrammar._matching_names: token {0}, pref={1}, suff={2}'
        #              ''.format(token, prefix, suffix))

        matching_names = list(self.alphabet)
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

