"""This module implements constants that are used inside the pitch,
duration and onset inference algorithm."""
from __future__ import print_function, unicode_literals, division

from builtins import range
from builtins import object
import operator

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


class InferenceEngineConstants(object):
    """This class stores the constants used for pitch inference."""

    ON_STAFFLINE_RATIO_TRHESHOLD = 0.2
    '''Magic number for determining whether a notehead is *on* a ledger
    line, or *next* to a ledger line: if the ratio between the smaller
    and larger vertical difference of (top, bottom) vs. l.l. (top, bottom)
    is smaller than this, it means the notehead is most probably *NOT*
    on the l.l. and is next to it.'''

    STAFFLINE_CLSNAME = 'staff_line'
    STAFFSPACE_CLSNAME = 'staff_space'
    STAFF_CLSNAME = 'staff'

    STAFF_CROPOBJECT_CLSNAMES = ['staff_line', 'staff_space', 'staff']

    STAFFLINE_CROPOBJECT_CLSNAMES = ['staff_line', 'staff_space']

    STAFFLINE_LIKE_CROPOBJECT_CLSNAMES = ['staff_line', 'ledger_line']

    STAFF_RELATED_CLSNAMES = {
        'staff_grouping',
        'measure_separator',
        'key_signature',
        'time_signature',
        'g-clef', 'c-clef', 'f-clef', 'other-clef',
    }

    STYSTEM_LEVEL_CLSNAMES = {
        'staff_grouping',
        'measure_separator',
    }

    NOTEHEAD_CLSNAMES = {
        'noteheadFull',
        'notehead-empty',
        'notehead-square',
        'notehead-square-full',
        'grace-notehead-full',
        'grace-notehead-empty',
    }

    NONGRACE_NOTEHEAD_CLSNAMES = {
        'noteheadFull',
        'notehead-empty',
    }

    CLEF_CLSNAMES = {
        'g-clef',
        'c-clef',
        'f-clef',
    }

    KEY_SIGNATURE_CLSNAMES = {
        'key_signature',
    }

    MEASURE_SEPARATOR_CLSNAMES = {
        'measure_separator',
    }

    FLAGS_CLSNAMES = {
        '8th_flag',
        '16th_flag',
        '32th_flag',
        '64th_and_higher_flag',
    }

    BEAM_CLSNAMES = {
        'beam',
    }

    FLAGS_AND_BEAMS ={
        '8th_flag',
        '16th_flag',
        '32th_flag',
        '64th_and_higher_flag',
        'beam',
    }

    ACCIDENTAL_CLSNAMES = {
        'sharp': 1,
        'flat': -1,
        'natural': 0,
        'double_sharp': 2,
        'double_flat': -2,
    }

    MIDI_CODE_RESIDUES_FOR_PITCH_STEPS = {
        0: 'C',
        1: 'C#',
        2: 'D',
        3: 'Eb',
        4: 'E',
        5: 'F',
        6: 'F#',
        7: 'G',
        8: 'Ab',
        9: 'A',
        10: 'Bb',
        11: 'B',
    }
    '''Simplified pitch naming.'''

    # The individual MIDI codes for for the unmodified steps.
    _fs = list(range(5, 114, 12))
    _cs = list(range(0, 121, 12))
    _gs = list(range(7, 116, 12))
    _ds = list(range(2, 110, 12))
    _as = list(range(9, 118, 12))
    _es = list(range(4, 112, 12))
    _bs = list(range(11, 120, 12))

    KEY_TABLE_SHARPS = {
        0: {},
        1: {i: 1 for i in _fs},
        2: {i: 1 for i in _fs + _cs},
        3: {i: 1 for i in _fs + _cs + _gs},
        4: {i: 1 for i in _fs + _cs + _gs + _ds},
        5: {i: 1 for i in _fs + _cs + _gs + _ds + _as},
        6: {i: 1 for i in _fs + _cs + _gs + _ds + _as + _es},
        7: {i: 1 for i in _fs + _cs + _gs + _ds + _as + _es + _bs},
    }

    KEY_TABLE_FLATS = {
        0: {},
        1: {i: -1 for i in _bs},
        2: {i: -1 for i in _bs + _es},
        3: {i: -1 for i in _bs + _es + _as},
        4: {i: -1 for i in _bs + _es + _as + _ds},
        5: {i: -1 for i in _bs + _es + _as + _ds + _gs},
        6: {i: -1 for i in _bs + _es + _as + _ds + _gs + _cs},
        7: {i: -1 for i in _bs + _es + _as + _ds + _gs + _cs + _fs},
    }

    # FROM clef --> TO clef. Imagine this on inline accidental delta
    CLEF_CHANGE_DELTA = {
        'g-clef': {
            'g-clef': 0,
            'c-clef': 6,
            'f-clef': 12,
        },
        'c-clef': {
            'g-clef': -6,
            'c-clef': 0,
            'f-clef': 6,
        },
        'f-clef': {
            'g-clef': -12,
            'c-clef': -6,
            'f-clef': 0,
        }
    }

    PITCH_STEPS = ['C', 'D', 'E', 'F', 'G', 'A', 'B',
                   'C', 'D', 'E', 'F', 'G', 'A', 'B']
    # Wrap around twice for easier indexing.

    ACCIDENTAL_CODES = {'sharp': '#', 'flat': 'b',
                        'double_sharp': 'x', 'double_flat': 'bb'}

    REST_CLSNAMES = {
        'whole_rest',
        'half_rest',
        'quarter_rest',
        '8th_rest',
        '16th_rest',
        '32th_rest',
        '64th_and_higher_rest',
        'multi-measure_rest',
    }

    MEAUSURE_LASTING_CLSNAMES = {
        'whole_rest',
        'multi-measure_rest',
        'repeat-measure',
    }

    TIME_SIGNATURES = {
        'time_signature',
    }

    TIME_SIGNATURE_MEMBERS = {
        'whole-time_mark',
        'alla_breve',
        'numeral_0',
        'numeral_1',
        'numeral_2',
        'numeral_3',
        'numeral_4',
        'numeral_5',
        'numeral_6',
        'numeral_7',
        'numeral_8',
        'numeral_9',
        'letter_other',
    }

    NUMERALS = {
        'numeral_0',
        'numeral_1',
        'numeral_2',
        'numeral_3',
        'numeral_4',
        'numeral_5',
        'numeral_6',
        'numeral_7',
        'numeral_8',
        'numeral_9',
    }

    @property
    def clsnames_affecting_onsets(self):
        """Returns a list of Node class names for objects
        that affect onsets. Assumes notehead and rest durations
        have already been given."""
        output = set()
        output.update(self.NONGRACE_NOTEHEAD_CLSNAMES)
        output.update(self.REST_CLSNAMES)
        output.update(self.MEASURE_SEPARATOR_CLSNAMES)
        output.update(self.TIME_SIGNATURES)
        output.add('repeat_measure')
        return output

    @property
    def clsnames_bearing_duration(self):
        """Returns the list of classes that actually bear duration,
        i.e. contribute to onsets of their descendants in the precedence
        graph."""
        output = set()
        output.update(self.NONGRACE_NOTEHEAD_CLSNAMES)
        output.update(self.REST_CLSNAMES)
        return output

    @staticmethod
    def interpret_numerals(numerals):
        """Returns the given numeral Node as a number, left to right."""
        for n in numerals:
            if n.clsname not in InferenceEngineConstants.NUMERALS:
                raise ValueError('Symbol {0} is not a numeral!'.format(n.uid))
        n_str = ''.join([n.clsname[-1]
                         for n in sorted(numerals, key=operator.attrgetter('left'))])
        return int(n_str)


_CONST = InferenceEngineConstants()
