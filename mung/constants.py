"""This module implements constants that are used inside the pitch,
duration and onset inference algorithm."""
import operator


class InferenceEngineConstants(object):
    """This class stores the constants used for pitch inference."""

    ON_STAFFLINE_RATIO_THRESHOLD = 0.2
    '''Magic number for determining whether a notehead is *on* a leger
    line, or *next* to a leger line: if the ratio between the smaller
    and larger vertical difference of (top, bottom) vs. l.l. (top, bottom)
    is smaller than this, it means the notehead is most probably *NOT*
    on the l.l. and is next to it.'''

    STAFFLINE_CLASS_NAME = 'staffLine'
    STAFFSPACE_CLASS_NAME = 'staffSpace'
    STAFF_CLASS_NAME = 'staff'
    LEGER_LINE_CLASS_NAME = 'legerLine'

    STAFF_CLASS_NAMES = [STAFFLINE_CLASS_NAME, STAFFSPACE_CLASS_NAME, STAFF_CLASS_NAME]
    STAFFLINE_CLASS_NAMES = [STAFFLINE_CLASS_NAME, STAFFSPACE_CLASS_NAME]

    STAFFLINE_LIKE_CLASS_NAMES = [STAFFLINE_CLASS_NAME, LEGER_LINE_CLASS_NAME]

    G_CLEF = 'gClef'
    C_CLEF = 'cClef'
    F_CLEF = 'fClef'

    STAFF_RELATED_CLASS_NAMES = {
        'staffGrouping',
        'measureSeparator',
        'keySignature',
        'timeSignature',
        G_CLEF,
        C_CLEF,
        F_CLEF
    }

    SYSTEM_LEVEL_CLASS_NAMES = {
        'staff_grouping',
        'measure_separator',
    }

    NOTEHEAD_CLASS_NAMES = {
        'noteheadFull',
        'noteheadHalf',
        'noteheadWhole',
        'noteheadFullSmall',
        'noteheadHalfSmall',
    }

    NONGRACE_NOTEHEAD_CLASS_NAMES = {
        'noteheadFull',
        'noteheadHalf',
        'noteheadWhole',
    }

    CLEF_CLASS_NAMES = {
        G_CLEF,
        C_CLEF,
        F_CLEF
    }

    KEY_SIGNATURE_CLASS_NAMES = {
        'keySignature',
    }

    MEASURE_SEPARATOR_CLASS_NAMES = {
        'measureSeparator',
    }

    FLAGS_CLASS_NAMES = {
        'flag8thUp',
        'flag8thDown',
        'flag16thUp',
        'flag16thDown',
        'flag32ndUp',
        'flag32ndDown',
        'flag64thUp',
        'flag64thDown',
    }

    BEAM_CLASS_NAMES = {
        'beam',
    }

    FLAGS_AND_BEAMS = set(list(FLAGS_CLASS_NAMES) + list(BEAM_CLASS_NAMES))

    ACCIDENTAL_CLASS_NAMES = {
        'accidentalSharp': 1,
        'accidentalFlat': -1,
        'accidentalNatural': 0,
        'accidentalDoubleSharp': 2,
        'accidentalDoubleFlat': -2,
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
        G_CLEF: {
            G_CLEF: 0,
            C_CLEF: 6,
            F_CLEF: 12,
        },
        C_CLEF: {
            G_CLEF: -6,
            C_CLEF: 0,
            F_CLEF: 6,
        },
        F_CLEF: {
            G_CLEF: -12,
            C_CLEF: -6,
            F_CLEF: 0,
        }
    }

    PITCH_STEPS = ['C', 'D', 'E', 'F', 'G', 'A', 'B',
                   'C', 'D', 'E', 'F', 'G', 'A', 'B']
    # Wrap around twice for easier indexing.

    ACCIDENTAL_CODES = {'accidentalSharp': '#', 'accidentalFlat': 'b',
                        'accidentalDoubleSharp': 'x', 'accidentalDoubleFlat': 'bb'}

    REST_CLASS_NAMES = {
        'restWhole',
        'restHalf',
        'restQuarter',
        'rest8th',
        'rest16th',
        'rest32nd',
        'rest64th',
        'multiMeasureRest',
    }

    MEAUSURE_LASTING_CLASS_NAMES = {
        'restWhole',
        'multiMeasureRest',
        'repeat1Bar',
    }

    TIME_SIGNATURES = {
        'timeSignature',
    }

    TIME_SIGNATURE_MEMBERS = {
        'timeSigCommon',
        'timeSigCutCommon',
        'numeral0',
        'numeral1',
        'numeral2',
        'numeral3',
        'numeral4',
        'numeral5',
        'numeral6',
        'numeral7',
        'numeral8',
        'numeral9',
    }

    NUMERALS = {
        'numeral0',
        'numeral1',
        'numeral2',
        'numeral3',
        'numeral4',
        'numeral5',
        'numeral6',
        'numeral7',
        'numeral8',
        'numeral9',
    }

    @property
    def classes_affecting_onsets(self):
        """Returns a list of Node class names for objects
        that affect onsets. Assumes notehead and rest durations
        have already been given."""
        output = set()
        output.update(self.NONGRACE_NOTEHEAD_CLASS_NAMES)
        output.update(self.REST_CLASS_NAMES)
        output.update(self.MEASURE_SEPARATOR_CLASS_NAMES)
        output.update(self.TIME_SIGNATURES)
        output.add('repeat1Bar')
        return output

    @property
    def classes_bearing_duration(self):
        """Returns the list of classes that actually bear duration,
        i.e. contribute to onsets of their descendants in the precedence
        graph."""
        output = set()
        output.update(self.NONGRACE_NOTEHEAD_CLASS_NAMES)
        output.update(self.REST_CLASS_NAMES)
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
