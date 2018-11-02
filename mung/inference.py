"""This module implements a class that..."""
from __future__ import print_function, unicode_literals, division
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
import collections
import copy
import logging
import os

import operator

from mung.node import bbox_dice
from mung.graph import group_staffs_into_systems, NotationGraph, NotationGraphError
from mung.inference_engine_constants import InferenceEngineConstants

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."

_CONST = InferenceEngineConstants()


class OnsetsInferenceStrategy(object):
    def __init__(self):
        self.permissive_desynchronization = True
        self.precedence_only_for_objects_connected_to_staff = True
        self.permissive = True


class PitchInferenceStrategy(object):
    def __init__(self):
        self.permissive = True


class PitchInferenceEngineState(object):
    """This class represents the state of the MIDI pitch inference
    engine during inference.

    Reading pitch is a stateful operations. One needs to remember
    how stafflines and staffspaces map to pitch codes. This is governed
    by two things:

    * The clef, which governs
    * The accidentals: key signatures and inline accidentals.

    Clef and key signature have unlimited scope which only changes when
    a new key signature is introduced (or the piece ends). The key
    signature affects all pitches in the given step class (F, C, G, ...)
    regardless of octave. Inline accidentals have scope until the next
    measure separator, and they are only valid within their own octave.

    The pitch inference algorithm is run for each staff separately.

    Base pitch representation
    -------------------------

    The base pitch corresponds to the pitch encoded by a notehead
    simply sitting on the given staffline/staffspace, without any
    regard to accidentals (key signature or inline). It is computed
    by *distance from center staffline* of a staff, with positive
    distance corresponding to going *up* and negative for going *down*
    from the center staffline.

    Accidentals representation
    --------------------------

    The accidentals are associated also with each staffline/staffspace,
    as counted from the current center. (This means i.a. that
    the octave periodicity is 7, not 12.)

    There are two kinds of accidentals based on scope: key signature,
    and inline. Inline accidentals are valid only up to the next
    measure_separator, while key signature accidentals are valid
    up until the key signature changes. Key signature accidentals
    also apply across all octaves, while inline accidentals only apply
    on the specific staffline.

    Note that inline accidentals may *cancel* key signature
    accidentals: they override the key signature when given.

    Key accidentals are given **mod 7**.

    Pitch inference procedure
    -------------------------

    Iterate through the relevant objects on a staff, sorted left-to-right
    by left edge.
    """
    def __init__(self):

        self.base_pitch = None
        '''The MIDI code corresponding to the middle staffline,
        without modification by key or inline accidentals.'''

        self.base_pitch_step = None
        '''The name of the base pitch: C, D, E, etc.'''

        self.base_pitch_octave = None
        '''The octave where the pitch resides. C4 = c', the middle C.'''

        self._current_clef = None
        '''Holds the clef Node that is currently valid.'''

        self._current_delta_steps = None
        '''Holds for each staffline delta step (i.e. staffline delta mod 7)
        the MIDI pitch codes.'''

        self._current_clef_delta_shift = 0
        '''If the clef is in a non-standard position, this number is added
        to the pitch computation delta.'''

        self.key_accidentals = {}
        self.inline_accidentals = {}

    def reset(self):
        self.base_pitch = None
        self._current_clef = None
        self._current_delta_steps = None
        self.key_accidentals = {}
        self.inline_accidentals = {}

    def __str__(self):
        lines = list()
        lines.append('Current pitch inference state:')
        lines.append('\tbase_pitch: {0}'.format(self.base_pitch))
        lines.append('\tbase_pitch_step: {0}'.format(self.base_pitch_step))
        lines.append('\tbase_pitch_octave: {0}'.format(self.base_pitch_octave))
        if self._current_clef is not None:
            lines.append('\t_current_clef: {0}'.format(self._current_clef.uid))
        else:
            lines.append('\t_current_clef: None')
        lines.append('\t_current_delta_steps: {0}'.format(self._current_delta_steps))
        lines.append('\t_current_clef_delta_shift: {0}'.format(self._current_clef_delta_shift))
        lines.append('\tkey_accidentals: {0}'.format(self.key_accidentals))
        lines.append('\tinline_accidentals: {0}'.format(self.inline_accidentals))
        return '\n'.join(lines)

    def init_base_pitch(self, clef=None, delta=0):
        """Initializes the base pitch while taking into account
        the displacement of the clef from its initial position."""
        self.init_base_pitch_default_staffline(clef)
        self._current_clef_delta_shift = -1 * delta

    def init_base_pitch_default_staffline(self, clef=None):
        """Based solely on the clef class name and assuming
        default stafflines, initialize the base pitch.
        By default, initializes as though given a g-clef."""

        # There should be a mechanism for clefs that are connected
        # directly to a staffline -- in non-standard positions
        # (mostly c-clefs, like page 03, but there is no reason
        #  to limit this to c-clefs).

        if (clef is None) or (clef.clsname == 'g-clef'):
            new_base_pitch = 71
            new_delta_steps = [0, 1, 2, 2, 1, 2, 2, 2]
            new_base_pitch_step = 6  # Index into pitch steps.
            new_base_pitch_octave = 4
        elif clef.clsname == 'f-clef':
            new_base_pitch = 50
            new_delta_steps = [0, 2, 1, 2, 2, 2, 1, 2]
            new_base_pitch_step = 1
            new_base_pitch_octave = 3
        elif clef.clsname == 'c-clef':
            new_base_pitch = 60
            new_delta_steps = [0, 2, 2, 1, 2, 2, 2, 1]
            new_base_pitch_step = 0
            new_base_pitch_octave = 4
        else:
            raise ValueError('Unrecognized clef clsname: {0}'
                             ''.format(clef.clsname))

        # Shift the key and inline accidental deltas
        # according to the change.
        if self._current_clef is not None:
            transposition_delta = _CONST.CLEF_CHANGE_DELTA[self._current_clef.clsname][clef.clsname]
            if transposition_delta != 0:
                new_key_accidentals = {
                    (d + transposition_delta) % 7: v
                    for d, v in list(self.key_accidentals.items())
                }
                new_inline_accidentals = {
                    d + transposition_delta: v
                    for d, v in list(self.inline_accidentals.items())
                }
                self.key_accidentals = new_key_accidentals
                self.inline_accidentals = new_inline_accidentals

        self.base_pitch = new_base_pitch
        self.base_pitch_step = new_base_pitch_step
        self.base_pitch_octave = new_base_pitch_octave
        self._current_clef = clef
        self._current_delta_steps = new_delta_steps

    def set_key(self, n_sharps=0, n_flats=0):
        """Initialize the staffline delta --> key accidental map.
        Currently works only on standard key signatures, where
        there are no repeating accidentals, no double sharps/flats,
        and the order of accidentals is the standard major/minor system.

        However, we can deal at least with key signatures that combine
        sharps and flats (if not more than 7), as seen e.g. in harp music.

        :param n_sharps: How many sharps are there in the key signature?

        :param n_flats: How many flats are there in the key signature?
        """
        if n_flats + n_sharps > 7:
            raise ValueError('Cannot deal with key signature that has'
                             ' more than 7 sharps + flats!')

        if self.base_pitch is None:
            raise ValueError('Cannot initialize key if base pitch is not known.')

        new_key_accidentals = {}

        # The pitches (F, C, G, D, ...) have to be re-cast
        # in terms of deltas, mod 7.
        if (self._current_clef is None) or (self._current_clef.clsname == 'g-clef'):
            deltas_sharp = [4, 1, 5, 2, 6, 3, 0]
            deltas_flat = [0, 3, 6, 2, 5, 1, 4]
        elif self._current_clef.clsname == 'c-clef':
            deltas_sharp = [3, 0, 4, 1, 5, 2, 6]
            deltas_flat = [6, 2, 5, 1, 4, 0, 3]
        elif self._current_clef.clsname == 'f-clef':
            deltas_sharp = [2, 6, 3, 0, 4, 1, 5]
            deltas_flat = [5, 1, 4, 0, 3, 6, 2]

        for d in deltas_sharp[:n_sharps]:
            new_key_accidentals[d] = 'sharp'
        for d in deltas_flat[:n_flats]:
            new_key_accidentals[d] = 'flat'

        self.key_accidentals = new_key_accidentals

    def set_inline_accidental(self, delta, accidental):
        self.inline_accidentals[delta] = accidental.clsname

    def reset_inline_accidentals(self):
        self.inline_accidentals = {}

    def accidental(self, delta):
        """Returns the modification, in MIDI code, corresponding
        to the staffline given by the delta."""
        pitch_mod = 0

        step_delta = delta % 7
        if step_delta in self.key_accidentals:
            if self.key_accidentals[step_delta] == 'sharp':
                pitch_mod = 1
            elif self.key_accidentals[step_delta] == 'double_sharp':
                pitch_mod = 2
            elif self.key_accidentals[step_delta] == 'flat':
                pitch_mod = -1
            elif self.key_accidentals[step_delta] == 'double_flat':
                pitch_mod = -2

        # Inline accidentals override key accidentals.
        if delta in self.inline_accidentals:
            if self.inline_accidentals[delta] == 'natural':
                logging.info('Natural at delta = {0}'.format(delta))
                pitch_mod = 0
            elif self.inline_accidentals[delta] == 'sharp':
                pitch_mod = 1
            elif self.inline_accidentals[delta] == 'double_sharp':
                pitch_mod = 2
            elif self.inline_accidentals[delta] == 'flat':
                pitch_mod = -1
            elif self.inline_accidentals[delta] == 'double_flat':
                pitch_mod = -2
        return pitch_mod

    def pitch(self, delta):
        """Given a staffline delta, returns the current MIDI pitch code.

        (This method is the main interface of the PitchInferenceEngineState.)

        :delta: Distance in stafflines + staffspaces from the middle staffline.
            Negative delta means distance *below*, positive delta is *above*.

        :returns: The MIDI pitch code for the given delta.
        """
        delta += self._current_clef_delta_shift

        # Split this into octave and step components.
        delta_step = delta % 7
        delta_octave = delta // 7

        # From the base pitch and clef:
        step_pitch = self.base_pitch \
                     + sum(self._current_delta_steps[:delta_step+1]) \
                     + (delta_octave * 12)
        accidental_pitch = self.accidental(delta)

        pitch = step_pitch + accidental_pitch

        if self._current_clef_delta_shift != 0:
            logging.info('PitchInferenceState: Applied clef-based delta {0},'
                         ' resulting delta was {1}, pitch {2}'
                         ''.format(self._current_clef_delta_shift,
                                   delta, pitch))

        return pitch

    def pitch_name(self, delta):
        """Given a staffline delta, returns the name of the corrensponding pitch."""
        delta += self._current_clef_delta_shift

        output_step = InferenceEngineConstants.PITCH_STEPS[(self.base_pitch_step + delta) % 7]
        output_octave = self.base_pitch_octave + ((delta + self.base_pitch_step) // 7)

        output_mod = ''
        accidental = self.accidental(delta)
        if accidental == 1:
            output_mod = InferenceEngineConstants.ACCIDENTAL_CODES['sharp']
        elif accidental == 2:
            output_mod = InferenceEngineConstants.ACCIDENTAL_CODES['double_sharp']
        elif accidental == -1:
            output_mod = InferenceEngineConstants.ACCIDENTAL_CODES['flat']
        elif accidental == 2:
            output_mod = InferenceEngineConstants.ACCIDENTAL_CODES['double_flat']

        return output_step + output_mod, output_octave


class PitchInferenceEngine(object):
    """The Pitch Inference Engine extracts MIDI from the notation
    graph. To get the MIDI, there are two streams of information
    that need to be combined: pitches and onsets, where the onsets
    are necessary both for ON and OFF events.

    Pitch inference is done through the ``infer_pitches()`` method.

    Onsets inference is done in two stages. First, the durations
    of individual notes (and rests) are computed, then precedence
    relationships are found and based on the precedence graph
    and durations, onset times are computed.

    Onset inference
    ---------------

    Onsets are computed separately by measure, which enables time
    signature constraint checking.

    (This can be implemented in the precedence graph structure,
    by (a) not allowing precedence edges to cross measure separators,
    (b) chaining measure separators, or it can be implemented
    directly in code. The first option is way more elegant.)

    Creating the precedence graph
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    * Get measure separators.
    * Chain measure separators in precedence relationships.
    * Group cropobjects by bins between measure separators.
    * For each staff participating in the current measure
      (as defined by the relevant measure separator outlinks):

        * Infer precedence between the participating notes & rests,
        * Attach the sources of the resulting DAG to the leftward
          measure_separator (if there is none, just leave them
          as sources).

    Invariants
    ^^^^^^^^^^

    * There is exactly one measure separator starting each measure,
      except for the first measure, which has none. That implies:
      when there are multiple disconnected barlines marking the interface
      of the same two measures within a system, they are joined under
      a single measure_separator anyway.
    * Staff groupings are correct, and systems are read top-down.

    """
    def __init__(self, strategy=PitchInferenceStrategy()):
        # Inference engine constants
        self._CONST = InferenceEngineConstants()

        # Static temp data from which the pitches are inferred
        self._cdict = {}

        self.strategy = strategy

        self.staves = None

        self.clefs = None
        self.clef_to_staff_map = None
        self.staff_to_clef_map = None

        self.key_signatures = None
        self.key_to_staff_map = None
        self.staff_to_key_map = None

        self.measure_separators = None
        self.staff_to_msep_map = None

        self.noteheads = None
        self.staff_to_noteheads_map = None

        # Dynamic temp data: things that change as the pitches are inferred.
        self.pitch_state = PitchInferenceEngineState()

        # Results
        self.pitches = None
        self.pitches_per_staff = None

        self.pitch_names = None
        self.pitch_names_per_staff = None

        # self.durations_beats = None
        # self.durations_beats_per_staff = None

    def reset(self):
        self.__init__()

    def infer_pitches(self, cropobjects, with_names=False):
        """The main workhorse for pitch inference.
        Gets a list of CropObjects and for each notehead-type
        symbol, outputs a MIDI code corresponding to the pitch
        encoded by that notehead.

        Notehead
        --------

        * Check for ties; if there is an incoming tie, apply
          the last pitch. (This is necessary because of ties
          that go across barlines and maintain inline accidentals.)
        * Determine its staffline delta from the middle staffline.
        * Check for inline accidentals, apply them to inference state.
        * Query pitch state with this staffline delta.

        Ties are problematic, because they may reach across
        staff breaks. This can only be resolved after all staves
        are resolved and assigned to systems, because until then,
        it is not clear which staff corresponds to which in the next
        system. Theoretically, this is near-impossible to resolve,
        because staves may not continue on the next system (e.g.,
        instruments that do not play for some time in orchestral scores),
        so simple staff counting is not foolproof. Some other matching
        mechanism has to be found, e.g. matching outgoing and incoming
        ties on the end and beginning of adjacent systems.

        Measure separator
        -----------------

        * Reset all inline accidentals to empty.

        Clef change
        -----------

        * Change base pitch
        * Recompute the key and inline signature delta indexes

        Key change
        ----------

        * Recompute key deltas

        :param with_names: If set, will return also a dict of
            objid --> pitch names (e.g., {123: 'F#3'}).

        :returns: A dict of ``objid`` to MIDI pitch code, with
            an entry for each (pitched) notehead. If ``with_names``
            is given, returns a tuple with the objid --> MIDI
            and objid --> pitch name dicts.

        """
        self._cdict = {c.objid: c for c in cropobjects}

        # Initialize pitch temp data.
        self._collect_symbols_for_pitch_inference(cropobjects)

        # Staff processing: this is where the inference actually
        # happens.
        self.pitches_per_staff = {}
        self.pitches = {}
        self.pitch_names_per_staff = {}
        self.pitch_names = {}
        # self.durations_beats = {}
        # self.durations_beats_per_staff = {}

        for staff in self.staves:
            self.process_staff(staff)
            self.pitches.update(self.pitches_per_staff[staff.objid])

        if with_names:
            return copy.deepcopy(self.pitches), copy.deepcopy(self.pitch_names)
        else:
            return copy.deepcopy(self.pitches)

    def process_staff(self, staff):

        self.pitches_per_staff[staff.objid] = {}
        self.pitch_names_per_staff[staff.objid] = {}

        # self.durations_beats_per_staff[staff.objid] = {}

        self.pitch_state.reset()
        self.pitch_state.init_base_pitch()

        queue = sorted(
                    self.staff_to_clef_map[staff.objid]
                    + self.staff_to_key_map[staff.objid]
                    + self.staff_to_msep_map[staff.objid]
                    + self.staff_to_noteheads_map[staff.objid],
                    key=lambda x: x.left)

        for q in queue:
            logging.info('process_staff(): processing object {0}-{1}'
                         ''.format(q.clsname, q.objid))
            if q.clsname in self._CONST.CLEF_CLSNAMES:
                self.process_clef(q)
            elif q.clsname in self._CONST.KEY_SIGNATURE_CLSNAMES:
                self.process_key_signature(q)
            elif q.clsname in self._CONST.MEASURE_SEPARATOR_CLSNAMES:
                self.process_measure_separator(q)
            elif q.clsname in self._CONST.NOTEHEAD_CLSNAMES:
                p, pn = self.process_notehead(q, with_name=True)
                self.pitches[q.objid] = p
                self.pitches_per_staff[staff.objid][q.objid] = p
                self.pitch_names[q.objid] = pn
                self.pitch_names_per_staff[staff.objid][q.objid] = pn

                ### DEBUG
                if q.objid in [131, 83, 89, 94]:
                    logging.info('PitchInferenceEngine: Processing notehead {0}'
                                 ''.format(q.uid))
                    logging.info('{0}'.format(self.pitch_state))

                # b = self.beats(q)
                # self.durations_beats[q.objid] = b
                # self.durations_beats_per_staff[staff.objid][q.objid] = b

        return self.pitches_per_staff[staff.objid]

    def process_notehead(self, notehead, with_name=False):
        """This is the main workhorse of the pitch inference engine.

        :param notehead: The notehead-class Node for which we
            want to infer pitch.

        :param with_name: If set, will return not only the MIDI pitch
            code, but the name of the encoded note (e.g., F#3) as well.
        """
        # Processing ties
        # ---------------
        ties = self.__children(notehead, 'tie')
        for t in ties:
            tied_noteheads = self.__parents(t, self._CONST.NOTEHEAD_CLSNAMES)

            # Corner cases: mistakes and staff breaks
            if len(tied_noteheads) > 2:
                raise ValueError('Tie {0}: joining together more than 2'
                                 ' noteheads!'.format(t.uid))
            if len(tied_noteheads) < 2:
                logging.warning('Tie {0}: only one notehead. Staff break?'
                                ''.format(t.uid))
                break

            left_tied_notehead = min(tied_noteheads, key=lambda x: x.left)
            if left_tied_notehead.objid != notehead.objid:
                try:
                    p = self.pitches[left_tied_notehead.objid]
                    if with_name:
                        pn = self.pitch_names[left_tied_notehead.objid]
                        return p, pn
                    else:
                        return p

                except KeyError:
                    raise KeyError('Processing tied notehead {0}:'
                                   ' preceding notehead {1} has no pitch!'
                                   ''.format(notehead.uid, left_tied_notehead.uid))

            # If the condition doesn't hold, then this is the leftward
            # note in the tie, and its pitch needs to be determined.

        # Obtain notehead delta
        # ---------------------
        delta = self.staffline_delta(notehead)

        # ### DEBUG
        # if notehead.objid == 200:
        #     logging.info('Notehead {0}: delta {1}'.format(notehead.uid, delta))
        #     logging.info('\tdelta_step: {0}'.format(delta % 7))
        #     logging.info('\tdelta_step pitch sum: {0}'
        #                  ''.format(sum(self.pitch_state._current_delta_steps[:(delta % 7)+1])))

        # Processing inline accidentals
        # -----------------------------
        accidentals = self.__children(notehead, self._CONST.ACCIDENTAL_CLSNAMES)

        if len(accidentals) > 0:

            # Sanity checks
            if len(accidentals) > 2:
                self.__warning_or_error('More than two accidentals attached to notehead'
                                        ' {0}'.format(notehead.uid))
            elif len(accidentals) == 2:
                naturals = [a for a in accidentals if a.clsname == 'natural']
                non_naturals = [a for a in accidentals if a.clsname != 'natural']
                if len(naturals) == 0:
                    self.__warning_or_error('More than one non-natural accidental'
                                            ' attached to notehead {0}'
                                            ''.format(notehead.uid))

                if len(non_naturals) == 0:
                    self.__warning_or_error('Two naturals attached to one notehead {0}'
                                            ''.format(notehead.uid))
                    self.pitch_state.set_inline_accidental(delta, naturals[0])
                else:
                    self.pitch_state.set_inline_accidental(delta, non_naturals[0])

            elif len(accidentals) == 1:
                self.pitch_state.set_inline_accidental(delta, accidentals[0])

        # Get the actual pitch
        # --------------------
        p = self.pitch_state.pitch(delta)

        ### DEBUG
        if notehead.objid in [131, 83, 89, 94]:
            logging.info('PitchInferenceEngine: results of pitch processing'
                         ' for notehead {0}'.format(notehead.uid))
            logging.info('\tties: {0}'.format(ties))
            logging.info('\taccidentals: {0}'.format(accidentals))
            logging.info('\tdelta: {0}'.format(delta))
            logging.info('\tpitch: {0}'.format(p))

        if with_name is True:
            pn = self.pitch_state.pitch_name(delta)
            return p, pn
        else:
            return p

    def staffline_delta(self, notehead):
        """Computes the staffline delta (distance from middle stafflines,
        measured in stafflines and staffspaces) for the given notehead
        (or any other symbol connected to a staffline/staffspace).
        Accounts for ledger lines.
        """
        current_staff = self.__children(notehead, ['staff'])[0]
        staffline_objects = self.__children(notehead,
                                            self._CONST.STAFFLINE_CROPOBJECT_CLSNAMES)

        # Ledger lines
        # ------------
        if len(staffline_objects) == 0:

            # Processing ledger lines:
            #  - count ledger lines
            lls = self.__children(notehead, 'ledger_line')
            n_lls = len(lls)
            if n_lls == 0:
                raise ValueError('Notehead with no staffline or staffspace,'
                                 ' but also no ledger lines: {0}'
                                 ''.format(notehead.uid))

            #  Determine: is notehead above or below staff?
            is_above_staff = (notehead.top < current_staff.top)

            #  Determine: is notehead on/next to (closest) ledger line?
            #    This needs to be done *after* we know whether the notehead
            #    is above/below staff: if the notehead is e.g. above,
            #    then it would be weird to find out it is in the
            #    mini-staffspace *below* the closest ledger line,
            #    signalling a mistake in the data.
            closest_ll = min(lls, key=lambda x: (x.top - notehead.top)**2 + (x.bottom - notehead.bottom)**2)

            # Determining whether the notehead is on a ledger
            # line or in the adjacent temp staffspace.
            # This uses a magic number, ON_STAFFLINE_RATIO_THRESHOLD.
            _on_ledger_line = True

            ### DEBUG!!!
            dtop, dbottom = 1, 1

            # Weird situation with notehead vertically *inside* bbox
            # of ledger line (could happen with slanted LLs and very small
            # noteheads).
            if closest_ll.top <= notehead.top <= notehead.bottom <= closest_ll.bottom:
                _on_ledger_line = True

            # No vertical overlap between LL and notehead
            elif closest_ll.top > notehead.bottom:
                _on_ledger_line = False
            elif notehead.top > closest_ll.bottom:
                _on_ledger_line = False

            # Complicated situations: overlap
            else:
                # Notehead "around" ledger line.
                if notehead.top < closest_ll.top <= closest_ll.bottom < notehead.bottom:
                    dtop = closest_ll.top - notehead.top
                    dbottom = notehead.bottom - closest_ll.bottom

                    if min(dtop, dbottom) / max(dtop, dbottom) \
                            < InferenceEngineConstants.ON_STAFFLINE_RATIO_TRHESHOLD:
                        _on_ledger_line = False

                        # Check orientation congruent with rel. to staff.
                        # If it is wrong (e.g., notehead mostly under LL
                        # but above staffline, and looks like off-LL),
                        # change back to on-LL.
                        if (dtop > dbottom) and not is_above_staff:
                            _on_ledger_line = True
                            logging.debug('Notehead in LL space with wrong orientation '
                                          'w.r.t. staff:'
                                          ' {0}'.format(notehead.uid))
                        if (dbottom > dtop) and is_above_staff:
                            _on_ledger_line = True
                            logging.debug('Notehead in LL space with wrong orientation '
                                          'w.r.t. staff:'
                                          ' {0}'.format(notehead.uid))

                # Notehead interlaced with ledger line, notehead on top
                elif notehead.top < closest_ll.top <= notehead.bottom <= closest_ll.bottom:
                    # dtop = closest_ll.top - notehead.top
                    # dbottom = max(notehead.bottom - closest_ll.top, 1)
                    # if float(dbottom) / float(dtop) \
                    #         < InferenceEngineConstants.ON_STAFFLINE_RATIO_TRHESHOLD:
                    _on_ledger_line = False

                # Notehead interlaced with ledger line, ledger line on top
                elif closest_ll.top <= notehead.top <= closest_ll.bottom < notehead.bottom:
                    # dtop = max(closest_ll.bottom - notehead.top, 1)
                    # dbottom = notehead.bottom - closest_ll.bottom
                    # if float(dtop) / float(dbottom) \
                    #         < InferenceEngineConstants.ON_STAFFLINE_RATIO_TRHESHOLD:
                    _on_ledger_line = False

                else:
                    raise ValueError('Strange notehead {0} vs. ledger line {1}'
                                     ' situation: bbox notehead {2}, LL {3}'
                                     ''.format(notehead.uid, closest_ll.uid,
                                               notehead.bounding_box,
                                               closest_ll.bounding_box))

            delta = (2 * n_lls - 1) + 5
            if not _on_ledger_line:
                delta += 1

            if not is_above_staff:
                delta *= -1

            ### DEBUG
            # if notehead.objid in [178]:
            #     logging.info('Notehead {0}: bbox {1}'.format(notehead.objid, notehead.bounding_box))
            #     logging.info('Closest LL objid: {0}'.format(closest_ll.objid))
            #     logging.info('no. of LLs: {0}'.format(n_lls))
            #     logging.info('Is above staff: {0}'.format(is_above_staff))
            #     logging.info('On ledger line: {0}'.format(_on_ledger_line))
            #     logging.info('Dtop: {0}, Dbottom: {1}'.format(dtop, dbottom))
            #     logging.info('Delta: {0}'.format(delta))

            return delta

        elif len(staffline_objects) == 1:
            current_staffline = staffline_objects[0]

            # Count how far from the current staffline we are.
            #  - Collect staffline objects from the current staff
            all_staffline_objects = self.__children(current_staff,
                                                    self._CONST.STAFFLINE_CROPOBJECT_CLSNAMES)

            #  - Determine their ordering, top to bottom
            sorted_staffline_objects = sorted(all_staffline_objects,
                                              key=lambda x: (x.top + x.bottom) / 2.)

            delta = None
            for i, s in enumerate(sorted_staffline_objects):
                if s.objid == current_staffline.objid:
                    delta = 5 - i

            if delta is None:
                raise ValueError('Notehead {0} attached to staffline {1},'
                                 ' which is however not a child of'
                                 ' the notehead\'s staff {2}!'
                                 ''.format(notehead.uid, current_staffline.uid,
                                           current_staff.uid))

            return delta

        else:
            raise ValueError('Notehead {0} attached to more than one'
                             ' staffline/staffspace!'.format(notehead.uid))

    def process_measure_separator(self, measure_separator):
        self.pitch_state.reset_inline_accidentals()

    def process_key_signature(self, key_signature):
        sharps = self.__children(key_signature, ['sharp'])
        flats = self.__children(key_signature, ['flat'])
        self.pitch_state.set_key(len(sharps), len(flats))

    def process_clef(self, clef):
        # Check for staffline children
        stafflines = self.__children(clef, clsnames=_CONST.STAFFLINE_CROPOBJECT_CLSNAMES)
        if len(stafflines) == 0:
            logging.info('Clef not connected to any staffline, assuming default'
                         ' position: {0}'.format(clef.uid))
            self.pitch_state.init_base_pitch(clef=clef)
        else:
            # Compute clef staffline delta from middle staffline.
            delta = self.staffline_delta(clef)
            logging.info('Clef {0}: computed staffline delta {1}'
                         ''.format(clef.uid, delta))
            self.pitch_state.init_base_pitch(clef=clef, delta=delta)

    def _collect_symbols_for_pitch_inference(self, cropobjects,
                                             ignore_nonstaff=True):
        """Extract all symbols from the document relevant for pitch
        inference and index them in the Engine's temp data structures."""
        graph = NotationGraph(cropobjects)

        # Collect staves.
        self.staves = [c for c in cropobjects if c.clsname == 'staff']
        logging.info('We have {0} staves.'.format(len(self.staves)))

        # Collect clefs and key signatures per staff.
        self.clefs = [c for c in cropobjects
                      if c.clsname in self._CONST.CLEF_CLSNAMES]
        if ignore_nonstaff:
            self.clefs = [c for c in self.clefs if graph.has_child(c, ['staff'])]

        self.key_signatures = [c for c in cropobjects
                               if c.clsname == 'key_signature']
        if ignore_nonstaff:
            self.key_signatures = [c for c in self.key_signatures
                                   if graph.has_child(c, ['staff'])]

        self.clef_to_staff_map = {}
        # There may be more than one clef per staff.
        self.staff_to_clef_map = collections.defaultdict(list)
        for c in self.clefs:
            # Assuming one staff per clef
            try:
                s = self.__children(c, ['staff'])[0]
            except (KeyError, ValueError):
                logging.warn('Clef {0} has no staff attached! Will not be'
                             ' part of pitch inference.'.format(c.uid))
                continue
            self.clef_to_staff_map[c.objid] = s
            self.staff_to_clef_map[s.objid].append(c)

        self.key_to_staff_map = {}
        # There may be more than one key signature per staff.
        self.staff_to_key_map = collections.defaultdict(list)
        for k in self.key_signatures:
            try:
                s = self.__children(k, ['staff'])[0]
            except KeyError:
                logging.warn('Key signature {0} has no staff attached! Will not be'
                             ' part of pitch inference.'.format(k.uid))
                continue
            self.key_to_staff_map[k.objid] = s
            self.staff_to_key_map[s.objid].append(k)

        # Collect measure separators.
        self.measure_separators = [c for c in cropobjects
                              if c.clsname == 'measure_separator']
        if ignore_nonstaff:
            self.measure_separators = [c for c in self.measure_separators
                                       if graph.has_child(c, ['staff'])]

        self.staff_to_msep_map = collections.defaultdict(list)
        for m in self.measure_separators:
            _m_staves = self.__children(m, ['staff'])
            # (Measure separators might belong to multiple staves.)
            for s in _m_staves:
                self.staff_to_msep_map[s.objid].append(m)
                # Collect accidentals per notehead.

        # Collect noteheads.
        self.noteheads = [c for c in cropobjects
                          if c.clsname in self._CONST.NOTEHEAD_CLSNAMES]
        if ignore_nonstaff:
            self.noteheads = [c for c in self.noteheads
                              if graph.has_child(c, ['staff'])]

        self.staff_to_noteheads_map = collections.defaultdict(list)
        for n in self.noteheads:
            s = self.__children(n, ['staff'])[0]
            self.staff_to_noteheads_map[s.objid].append(n)

    def __children(self, c, clsnames):
        """Retrieve the children of the given Node ``c``
        that have class in ``clsnames``."""
        return [self._cdict[o] for o in c.outlinks
                if self._cdict[o].clsname in clsnames]

    def __parents(self, c, clsnames):
        """Retrieve the parents of the given Node ``c``
        that have class in ``clsnames``."""
        return [self._cdict[i] for i in c.inlinks
                if self._cdict[i].clsname in clsnames]

    def __warning_or_error(self, message):
        if self.strategy.permissive:
            logging.warn(message)
        else:
            raise ValueError(message)


class OnsetsInferenceEngine(object):

    def __init__(self, cropobjects, strategy=OnsetsInferenceStrategy()):
        """Initialize the onset inference engine with the full Node
        list in a document."""
        self._CONST = InferenceEngineConstants()
        self._cdict = {c.objid: c for c in cropobjects}

        self.strategy = strategy

    def durations(self, cropobjects, ignore_modifiers=False):
        """Returns a dict that contains the durations (in beats)
        of all CropObjects that should be associated with a duration.
        The dict keys are ``objid``.

        :param ignore_modifiers: If set, will ignore duration dots,
            tuples, and other potential duration modifiers when computing
            the durations. Effectively, this gives you classes that
            correspond to note(head) type: whole (4.0), half (2.0),
            quarter (1.0), eighth (0.5), etc.
        """
        # Generate & return the durations dictionary.
        _relevant_clsnames = self._CONST.clsnames_bearing_duration
        d_cropobjects = [c for c in cropobjects
                         if c.clsname in _relevant_clsnames]

        durations = {c.objid: self.beats(c,
                                         ignore_modifiers=ignore_modifiers)
                     for c in d_cropobjects}
        return durations

    def beats(self, cropobject, ignore_modifiers=False):
        if cropobject.clsname in self._CONST.NOTEHEAD_CLSNAMES:
            return self.notehead_beats(cropobject,
                                       ignore_modifiers=ignore_modifiers)
        elif cropobject.clsname in self._CONST.REST_CLSNAMES:
            return self.rest_beats(cropobject,
                                   ignore_modifiers=ignore_modifiers)
        else:
            raise ValueError('Cannot compute beats for object {0} of class {1};'
                             ' beats only available for notes and rests.'
                             ''.format(cropobject.uid, cropobject.clsname))

    def notehead_beats(self, notehead, ignore_modifiers=False):
        """Retrieves the duration for the given notehead, in beats.

        It is possible that the notehead has two stems.
        In that case, we return all the possible durations:
        usually at most two, but if there is a duration dot, then
        there can be up to 4 possibilities.

        Grace notes currently return 0 beats.

        :param ignore_modifiers: If given, will ignore all duration
            modifiers: Duration dots, tuples, and other potential duration
            modifiers when computing the durations. Effectively, this
            gives you classes that correspond to note(head) type:
            whole (4.0), half (2.0), quarter (1.0), eighth (0.5), etc.

        :returns: A list of possible durations for the given notehead.
            Mostly its length is just 1; for multi-stem noteheads,
            you might get more.
        """
        beat = [0]

        stems = self.__children(notehead, ['stem'])
        flags_and_beams = self.__children(
            notehead,
            InferenceEngineConstants.FLAGS_AND_BEAMS)

        if notehead.clsname.startswith('grace-notehead'):
            logging.warn('Notehead {0}: Grace notes get zero duration!'
                         ''.format(notehead.uid))
            beat = [0]

        elif len(stems) > 1:
            logging.warn('Inferring duration for multi-stem notehead: {0}'
                         ''.format(notehead.uid))
            beat = self.process_multistem_notehead(notehead)
            if len(beat) > 1:
                self.__warning_or_error('Cannot deal with multi-stem notehead'
                                        ' where multiple durations apply.')
                beat = [max(beat)]

        elif notehead.clsname == 'notehead-empty':
            if len(flags_and_beams) != 0:
                raise ValueError('Notehead {0} is empty, but has {1} flags and beams!'
                                 ''.format(notehead.uid))

            if len(stems) == 0:
                beat = [4]
            else:
                beat = [2]

        elif notehead.clsname == 'notehead-full':
            if len(stems) == 0:
                self.__warning_or_error('Full notehead {0} has no stem!'.format(notehead.uid))

            beat = [0.5**len(flags_and_beams)]

        else:
            raise ValueError('Notehead {0}: unknown clsname {1}'
                             ''.format(notehead.uid, notehead.clsname))

        if not ignore_modifiers:
            duration_modifier = self.compute_duration_modifier(notehead)
            beat = [b * duration_modifier for b in beat]

        if len(beat) > 1:
            logging.warning('Notehead {0}: more than 1 duration: {1}, choosing first'
                            ''.format(notehead.uid, beat))
        return beat[0]

    def compute_duration_modifier(self, notehead):
        """Computes the duration modifier (multiplicative, in beats)
        for the given notehead (or rest) from the tuples and duration dots.

        Can handle duration dots within tuples.

        Cannot handle nested/multiple tuples.
        """
        duration_modifier = 1
        # Dealing with tuples:
        tuples = self.__children(notehead, ['tuple'])
        if len(tuples) > 1:
            raise ValueError('Notehead {0}: Cannot deal with more than one tuple'
                             ' simultaneously.'.format(notehead.uid))
        if len(tuples) == 1:
            tuple = tuples[0]

            # Find the number in the tuple.
            numerals = sorted([self._cdict[o] for o in tuple.outlinks
                               if self._cdict[o].clsname.startswith('numeral')],
                              key=lambda x: x.left)
            # Concatenate numerals left to right.
            tuple_number = int(''.join([num.clsname[-1] for num in numerals]))

            # Last note in tuple should get complementary duration
            # to sum to a whole. Otherwise, playing brings slight trouble.

            if tuple_number == 2:
                # Duola makes notes *longer*
                duration_modifier = 3 / 2
            elif tuple_number == 3:
                duration_modifier = 2 / 3
            elif tuple_number == 4:
                # This one also makes notes longer
                duration_modifier = 4 / 3
            elif tuple_number == 5:
                duration_modifier = 4 / 5
            elif tuple_number == 6:
                # Most often done for two consecutive triolas,
                # e.g. 16ths with a 6-tuple filling one beat
                duration_modifier = 2 / 3
            elif tuple_number == 7:
                # Here we get into trouble, because this one
                # can be both 4 / 7 (7 16th in a beat)
                # or 8 / 7 (7 32nds in a beat).
                # In the same vein, we cannot resolve higher
                # tuples unless we establish precedence/simultaneity.
                logging.warn('Cannot really deal with higher tuples than 6.')
                # For MUSCIMA++ specifically, we can cheat: there is only one
                # septuple, which consists of 7 x 32rd in 1 beat, so they
                # get 8 / 7.
                logging.warn('MUSCIMA++ cheat: we know there is only 7 x 32rd in 1 beat'
                             ' in page 14.')
                duration_modifier = 8 / 7
            elif tuple_number == 10:
                logging.warn('MUSCIMA++ cheat: we know there is only 10 x 32rd in 1 beat'
                             ' in page 04.')
                duration_modifier = 4 / 5
            else:
                raise NotImplementedError('Notehead {0}: Cannot deal with tuple '
                                          'number {1}'.format(notehead.uid,
                                                              tuple_number))

        # Duration dots
        ddots = self.__children(notehead, ['duration-dot'])
        dot_duration_modifier = 1
        for i, d in enumerate(ddots):
            dot_duration_modifier += 1 / (2 ** (i + 1))
        duration_modifier *= dot_duration_modifier

        return duration_modifier

    def rest_beats(self, rest, ignore_modifiers=False):
        """Compute the duration of the given rest in beats.

        :param ignore_modifiers: If given, will ignore all duration
            modifiers: Duration dots, tuples, and other potential duration
            modifiers when computing the durations. Effectively, this
            gives you classes that correspond to note(head) type:
            whole (4.0), half (2.0), quarter (1.0), eighth (0.5), etc.
            Also ignores deriving duration from the time signature
            for whole rests.

        """
        rest_beats_dict = {'whole_rest': 4,   # !!! We should find the Time Signature.
                           'half_rest': 2,
                           'quarter_rest': 1,
                           '8th_rest': 0.5,
                           '16th_rest': 0.25,
                           '32th_rest': 0.125,
                           '64th_and_higher_rest': 0.0625,
                           # Technically, these two should just apply time sig.,
                           # but the measure-factorized precedence graph
                           # means these durations never have sounding
                           # descendants anyway:
                           'multi-measure_rest': 4,
                           'repeat-measure': 4,
                           }

        try:
            base_rest_duration = rest_beats_dict[rest.clsname]

        except KeyError:
            raise KeyError('Symbol {0}: Unknown rest type {1}!'
                           ''.format(rest.uid, rest.clsname))

        # Process the whole rest:
        #  - if it is the only symbol in the measure, it should take on
        #    the duration of the current time signature.
        #  - if it is not the only symbol in the measure, it takes 4 beats
        #  - Theoretically, it could perhaps take e.g. 6 beats in weird situations
        #    in a 6/2 time signature, but we don't care about this for now.
        #
        # If there is no leftward time signature, we need to infer the time
        # sig from the other symbols. This necessitates two-pass processing:
        # first get all available durations, then guess the time signatures
        # (technically this might also be needed for each measure).
        if (rest.clsname in _CONST.MEAUSURE_LASTING_CLSNAMES) and not ignore_modifiers:
            base_rest_duration = self.measure_lasting_beats(rest)
            beat = [base_rest_duration]  # Measure duration should never be ambiguous.

        elif not ignore_modifiers:
            duration_modifier = self.compute_duration_modifier(rest)
            beat = [base_rest_duration * duration_modifier]

        else:
            beat = [base_rest_duration]

        if len(beat) > 1:
            logging.warning('Rest {0}: more than 1 duration: {1}, choosing first'
                            ''.format(rest.uid, beat))
        return beat[0]

    def measure_lasting_beats(self, cropobject):
        """Find the duration of an object that lasts for an entire measure
        by interpreting the time signature valid for the given point in
        the score.

        If any assumption is broken, will return the default measure duration:
        4 beats."""
        # Find rightmost preceding time signature on the staff.
        graph = NotationGraph(list(self._cdict.values()))

        # Find current time signature
        staffs = graph.children(cropobject, classes=[_CONST.STAFF_CLSNAME])

        if len(staffs) == 0:
            logging.warning('Interpreting object {0} as measure-lasting, but'
                            ' it is not attached to any staff! Returning default: 4'
                            ''.format(cropobject.uid))
            return 4

        if len(staffs) > 1:
            logging.warning('Interpreting object {0} as measure-lasting, but'
                            ' it is connected to more than 1 staff: {1}'
                            ' Returning default: 4'
                            ''.format(cropobject.uid, [s.uid for s in staffs]))
            return 4

        logging.info('Found staffs: {0}'.format([s.uid for s in staffs]))

        staff = staffs[0]
        time_signatures = graph.ancestors(staff, classes=_CONST.TIME_SIGNATURES)

        logging.info('Time signatures: {0}'.format([t.uid for t in time_signatures]))

        applicable_time_signatures = sorted([t for t in time_signatures
                                             if t.left < cropobject.left],
                                            key=operator.attrgetter('left'))
        logging.info('Applicable time signatures: {0}'.format([t.uid for t in time_signatures]))

        if len(applicable_time_signatures) == 0:
            logging.warning('Interpreting object {0} as measure-lasting, but'
                            ' there is no applicable time signature. Returnig'
                            ' default: 4'.format(cropobject.uid))
            return 4

        valid_time_signature = applicable_time_signatures[-1]
        beats = self.interpret_time_signature(valid_time_signature)
        return beats

    def process_multistem_notehead(self, notehead):
        """Attempts to recover the duration options of a multi-stem note."""
        stems = self.__children(notehead, ['stem'])
        flags_and_beams = self.__children(
            notehead,
            InferenceEngineConstants.FLAGS_AND_BEAMS)

        if len(flags_and_beams) == 0:
            if notehead.clsname == 'notehead-full':
                return [1]
            elif notehead.clsname == 'notehead-empty':
                return [2]

        if notehead.clsname == 'notehead-empty':
            raise NotationGraphError('Empty notehead with flags and beams: {0}'
                                     ''.format(notehead.uid))

        n_avg_x = notehead.top + (notehead.bottom - notehead.top) / 2.0
        print('Notehead {0}: avg_x = {1}'.format(notehead.uid, n_avg_x))
        f_and_b_above = []
        f_and_b_below = []
        for c in flags_and_beams:
            c_avg_x = c.top + (c.bottom - c.top) / 2.0
            print('Beam/flag {0}: avg_x = {1}'.format(c.uid, c_avg_x))
            if c_avg_x < n_avg_x:
                f_and_b_above.append(c)
                print('Appending above')
            else:
                f_and_b_below.append(c)
                print('Appending below')

        beat_above = 0.5**len(f_and_b_above)
        beat_below = 0.5**len(f_and_b_below)

        if beat_above != beat_below:
            raise NotImplementedError('Cannot deal with multi-stem note'
                                      ' that has different pre-modification'
                                      ' durations: {0} vs {1}'
                                      '{2}'.format(beat_above, beat_below, notehead.uid))

        beat = [beat_above]

        tuples = self.__children(notehead, ['tuple'])
        if len(tuples) % 2 != 0:
            raise NotImplementedError('Cannot deal with multi-stem note'
                                      ' that has an uneven number of tuples:'
                                      ' {0}'.format(notehead.uid))

        duration_modifier = self.compute_duration_modifier(notehead)
        beat = [b * duration_modifier for b in beat]

        return beat

    ##########################################################################
    # Onsets inference
    def infer_precedence_from_annotations(self, cropobjects):
        """Infer precedence graph based solely on the "green lines"
        in MUSCIMA++ annotation: precedence edges. These are encoded
        in the data as inlink/outlink lists
        in ``cropobject.data['precedence_inlinks']``,
        aand ``cropobject.data['precedence_outlinks']``.

        :param cropobjects: A list of CropObjects, not necessarily
            only those that participate in the precedence graph.

        :return: The list of source nodes of the precedence graph.
        """
        _relevant_clsnames = self._CONST.clsnames_bearing_duration
        p_cropobjects = [c for c in cropobjects
                         if c.clsname in _relevant_clsnames]

        if self.strategy.precedence_only_for_objects_connected_to_staff:
            p_cropobjects = [c for c in p_cropobjects
                             if len(self.__children(c, ['staff'])) > 0]

        durations = {c.objid: self.beats(c) for c in p_cropobjects}

        p_nodes = {}
        for c in p_cropobjects:
            p_node = PrecedenceGraphNode(objid=c.objid,
                                         cropobject=c,
                                         inlinks=[],
                                         outlinks=[],
                                         duration=durations[c.objid],
                                         )
            p_nodes[c.objid] = p_node

        for c in p_cropobjects:
            inlinks = []
            outlinks = []
            if 'precedence_inlinks' in c.data:
                inlinks = c.data['precedence_inlinks']
            if 'precedence_outlinks' in c.data:
                outlinks = c.data['precedence_outlinks']
            p_node = p_nodes[c.objid]
            p_node.outlinks = [p_nodes[o] for o in outlinks]
            p_node.inlinks = [p_nodes[i] for i in inlinks]

        # Join staves/systems!

        # ...systems:
        systems = group_staffs_into_systems(cropobjects,
                                            use_fallback_measure_separators=True)
        # _cdict = {c.objid: c for c in cropobjects}
        # staff_groups = [c for c in cropobjects
        #                 if c.clsname == 'staff_grouping']
        #
        # if len(staff_groups) != 0:
        #     staffs_per_group = {c.objid: [_cdict[i] for i in c.outlinks
        #                                   if _cdict[i].clsname == 'staff']
        #                         for c in staff_groups}
        #     # Build hierarchy of staff_grouping based on inclusion.
        #     outer_staff_groups = []
        #     for sg in staff_groups:
        #         sg_staffs = staffs_per_group[sg.objid]
        #         is_outer = True
        #         for other_sg in staff_groups:
        #             if sg.objid == other_sg.objid: continue
        #             other_sg_staffs = staffs_per_group[other_sg.objid]
        #             if len([s for s in sg_staffs
        #                     if s not in other_sg_staffs]) == 0:
        #                 is_outer = False
        #         if is_outer:
        #             outer_staff_groups.append(sg)
        #     #
        #     # outer_staff_groups = [c for c in staff_groups
        #     #                       if len([_cdict[i] for i in c.inlinks
        #     #                               if _cdict[i].clsname == 'staff_group']) == 0]
        #     systems = [[c for c in cropobjects
        #                 if (c.clsname == 'staff') and (c.objid in sg.outlinks)]
        #                for sg in outer_staff_groups]
        # else:
        #     systems = [[c] for c in cropobjects if c.clsname == 'staff']

        if len(systems) == 1:
            logging.info('Single-system score, no staff chaining needed.')
            source_nodes = [n for n in list(p_nodes.values()) if len(n.inlinks) == 0]
            return source_nodes

        # Check all systems same no. of staffs
        _system_lengths = [len(s) for s in systems]
        if len(set(_system_lengths)) > 1:
            raise ValueError('Cannot deal with variable number of staffs'
                             ' w.r.t. systems! Systems: {0}'.format(systems))

        staff_chains = [[] for _ in systems[0]]
        for system in systems:
            for i, staff in enumerate(system):
                staff_chains[i].append(staff)

        # Now, join the last --> first nodes within chains.

        # - Assign objects to staffs
        objid2staff = {}
        for c in cropobjects:
            staffs = self.__children(c, ['staff'])
            if len(staffs) == 1:
                objid2staff[c.objid] = staffs[0].objid

        # - Assign staffs to sink nodes
        sink_nodes2staff = {}
        staff2sink_nodes = collections.defaultdict(list)
        for node in list(p_nodes.values()):
            if len(node.outlinks) == 0:
                try:
                    staff = self.__children(node.obj, ['staff'])[0]
                except IndexError:
                    logging.error('Object {0} is a sink node in the precedence graph, but has no staff!'
                                  ''.format(node.obj.objid))
                    raise
                sink_nodes2staff[node.obj.objid] = staff.objid
                staff2sink_nodes[staff.objid].append(node)

        # Note that this means you should never have a sink node
        # unless it's at the end of the staff. All notes have to lead
        # somewhere. This is suboptimal; we should filter out non-maximal
        # sink nodes. But since we do not know whether the sink nodes
        # are maximal until we are done inferring onsets, we have to stick
        # with this.
        # The alternative is to only connect to the next staff the *rightmost*
        # sink node. This risks *not* failing if the sink nodes of a staff
        # are not synchronized properly.

        # - Assign staffs to source nodes
        source_nodes2staff = {}
        staff2source_nodes = collections.defaultdict(list)
        for node in list(p_nodes.values()):
            if len(node.inlinks) == 0:
                staff = self.__children(node.obj, ['staff'])[0]
                source_nodes2staff[node.obj.objid] = staff.objid
                staff2source_nodes[staff.objid].append(node)

        # - For each staff chain, link the sink nodes of the prev
        #   to the source nodes of the next staff.
        for staff_chain in staff_chains:
            staffs = sorted(staff_chain, key=lambda x: x.top)
            for (s1, s2) in zip(staffs[:-1], staffs[1:]):
                sinks = staff2sink_nodes[s1.objid]
                sources = staff2source_nodes[s2.objid]
                for sink in sinks:
                    for source in sources:
                        sink.outlinks.append(source)
                        source.inlinks.append(sink)

        source_nodes = [n for n in list(p_nodes.values()) if len(n.inlinks) == 0]
        return source_nodes

    def infer_precedence(self, cropobjects):
        """This is the most complex part of onset computation.

        The output of this method is a **precedence graph**. The precedence
        graph is a Directed Acyclic Graph (DAG) consisting of
        :class:`PrecedenceGraphNode` objects. Each node represents some
        musical concept that participates in establishing the onsets
        by having a *duration*. The invariant of the graph is that
        the onset of a node is the sum of the durations on each of its
        predecessor paths to a root node (which has onset 0).

        Not all nodes necessarily have nonzero duration (although these
        nodes can technically be factored out).

        Once the precedence graph is formed, then a breadth-first search
        (rather than DFS, to more easily spot/resolve conflicts at multi-source
        precedence graph nodes) simply accumulates durations.
        Conflicts can be resolved through failing (currently implemented),
        or looking up possible errors in assigning durations and attempting
        to fix them.

        Forming the precedence graph itself is difficult, because
        of polyphonic (and especially pianoform) music. Practically the only
        actual constraint followed throughout music is that *within a voice*
        notes are read left-to-right. The practice of aligning e.g. whole
        notes in an outer voice to the beginning of the bar rather than
        to the middle took over only cca. 1800 or later.

        An imperfect but overwhelmingly valid constraint is that notes taking
        up a certain proportion of the measure are not written to the *right*
        of the proportional horizontal span in the measure corresponding
        to their duration in time. However, this is *not* uniform across
        the measure: e.g., if the first voice is 2-8-8-8-8 and the second
        is 2-2, then the first half can be very narrow and the second
        quite wide, with the second lower-voice half-note in the middle
        of that part. However, the *first* lower-voice half-note will
        at least *not* be positioned in the horizontal span where
        the 8th notes in the upper voice are.

        Which CropObjects participate in the precedence graph?
        ------------------------------------------------------

        We directly derive precedence graph nodes from the following
        CropObjects:

        * Noteheads: empty, full, and grace noteheads of both kinds, which
          are assigned duration based on their type (e.g., quarter, 16th, etc.)
          and then may be further modified by duration dots and/or tuples.
        * Rests of all kinds, which get duration via a simple table based
          on the rest class and tuple/dot modification.
        * Measure separators, which get a duration of 0.

        The assumption of our symbol classes is that there are no rests
        shorter than 64th.

        Furthermore, we add synthetic nodes representing:

        * Root measure separator, with duration 0 **and** onset 0,
          which initializes the onset computations along the graph
        * Measure nodes, with durations derived from time signatures
          valid for the given measures.

        Constructing the precedence graph
        ---------------------------------

        We factor the precedence graph into measures, and then infer precedence
        for each measure separately, in order to keep the problem tractable
        and in order for errors not to propagate too far. The inference
        graph construction algorithm is therefore split into two steps:

        * Construct the "spine" of the precedence graph from measure nodes,
        * Construct the single-measure precedence subgraphs (further factored
          by staff).

        The difficulties lie primarily in step 2.

        (Note that ties are currently disregarded: the second note
        of the tie normally gets an onset. After all, conceptually,
        it is a separate note with an onset, it just does not get played.)

        Precedence graph spine
        ^^^^^^^^^^^^^^^^^^^^^^

        The **spine** of the precedence graph is a single path of alternating
        ``measure_separator`` and ``measure`` nodes. ``measure_separator``
        nodes are constructed from the CropObjects, and ``measure`` nodes
        are created artificially between consecutive ``measure_separator``
        nodes. The measure separator nodes have a duration of 0, while
        the duration of the measure nodes is inferred from the time signature
        valid for that measure. An artificial root measure_separator node
        is created to serve as the source of the entire precedence graph.

        Thus, the first part of the algorithm is, at a high level:

        * Order measure separators,
        * Assign time signatures to measures and compute measure durations
          from time signatures.

        **Gory details:** In step 1, we assume that systems are ordered
        top-down in time, that all systems are properly grouped using
        ``staff_grouping`` symbols, that measure separators are strictly
        monotonous (i.e., the same subset of possible onsets belongs to
        the i-th measure on each staff, which is an assumption that does
        *not* hold for isorhythmic motets and basically anything pre-16th
        century).

        In step 2, we assume that time signatures are always written within
        the measure that *precedes* the first measure for which they are
        valid, with the exception of the first time signature on the system.

        We also currently assume that a given measure has the same number
        of beats across all staves within a system (so: no polytempi for now).

        Measure subgraphs
        ^^^^^^^^^^^^^^^^^

        There are again two high-level steps:

        * Assign other onset-carrying objects (noteheads and rests)
          to measures, to prepare the second phase that iterates over
          these groups per measure (and staff).
        * For each measure group, compute the subgraph and attach
          its sources to the preceding measure separator node.

        The first one can be resolved easily by looking at (a) staff
        assignment, (b) horizontal position with respect to measure
        separators. Noting that long measure separators might not
        really be straight, we use the intersection of the separator
        with the given staff.

        The second step is the difficult one. We describe the algorithm
        for inferring precedence, simultaneity span minimization,
        in a separate section.


        Simultaneity span minimization
        ------------------------------

        Inferring precedence in polyphonic music is non-trivial, especially
        if one wants to handle handwritten music, and even more so when
        extending the scope before the 1800s. We infer precedence using
        the principle that notes which are supposed to be played together
        should be as close to each other horizontally as possible: from
        all the possible precedence assignments that fulfill notation
        rule constraints, choose the one which minimizes the horizontal
        span assigned to each unit of musical time in the bar.

        The algorithm is initialized as follows:

        * Determine the shortest subdivision of the measure (in beats)
          which has to be treated independently. This generally corresponds
          to the shortest note in the measure.
        * Initialize the assignment table: for each onset-carrying object,
          we will assign it to one of the time bins.

        There are some rules of music notation that we use to prune the space
        of possible precedence assignments by associating the notes (or rests)
        into blocks:

        * Beamed groups without intervening rests
        * Tied note pairs
        * Notes that share a stem
        * Notes within a tuple

        Rests within beamed groups (e.g., 8th - 8th_rest - 8th) are a problem.
        A decision needs to be made whether the rest does belong to the group
        or not.

        """

        if not self.measure_separators:
            self._collect_symbols_for_pitch_inference(cropobjects)

        measure_separators = [c for c in cropobjects
                              if c.clsname in self._CONST.MEASURE_SEPARATOR_CLSNAMES]

        ######################################################################
        # An important feature of measure-factorized onset inference
        # instead of going left-to-right per part throughout is resistance
        # to staves appearing & disappearing on line breaks (e.g. orchestral
        # scores). Measures are (very, very often) points of synchronization
        #  -- after all, that is their purpose.

        # We currently DO NOT aim to process renaissance & medieval scores:
        # especially motets may often have de-synchronized measure separators.

        # Add the relationships between the measure separator nodes.
        #  - Get staves to which the mseps are connected
        msep_staffs = {m.objid: self.__children(m, ['staff'])
                       for m in measure_separators}
        #  - Sort first by bottom-most staff to which the msep is connected
        #    to get systems
        #  - Sort left-to-right within systems to get final ordering of mseps
        ordered_mseps = sorted(measure_separators,
                               key=lambda m: (max([s.bottom
                                                   for s in msep_staffs[m.objid]]),
                                              m.left))
        ordered_msep_nodes = [PrecedenceGraphNode(cropobject=m,
                                                  inlinks=[],
                                                  outlinks=[],
                                                  onset=None,
                                                  duration=0)
                              for m in ordered_mseps]

        # Add root node: like measure separator, but for the first measure.
        # This one is the only one which is initialized with onset,
        # with the value onset=0.
        root_msep = PrecedenceGraphNode(objid=-1,
                                        cropobject=None,
                                        inlinks=[], outlinks=[],
                                        duration=0,
                                        onset=0)

        # Create measure bins. i-th measure ENDS at i-th ordered msep.
        # We assume that every measure has a rightward separator.
        measures = [(None, ordered_mseps[0])] + [(ordered_mseps[i], ordered_mseps[i+1])
                                                 for i in range(len(ordered_mseps) - 1)]
        measure_nodes = [PrecedenceGraphNode(objid=None,
                                             cropobject=None,
                                             inlinks=[root_msep],
                                             outlinks=[ordered_msep_nodes[0]],
                                             duration=0,  # Durations will be filled in
                                             onset=None)] + \
                        [PrecedenceGraphNode(objid=None,
                                             cropobject=None,
                                             inlinks=[ordered_msep_nodes[i+1]],
                                             outlinks=[ordered_msep_nodes[i+2]],
                                             duration=0,  # Durations will be filled in
                                             onset=None)
                         for i in range(len(ordered_msep_nodes) - 2)]
        #: A list of PrecedenceGraph nodes. These don't really need any Node
        #  or objid, they are just introducing through their duration the offsets
        #  between measure separators (mseps have legit 0 duration, so that they
        #  do not move the notes in their note descendants).
        #  The list is already ordered.

        # Add measure separator inlinks and outlinks.
        for m_node in measure_nodes:
            r_sep = m_node.outlinks[0]
            r_sep.inliks.append(m_node)
            if len(m_node.inlinks) > 0:
                l_sep = m_node.inlinks[0]
                l_sep.outlinks.append(m_node)

        # Finally, hang the first measure on the root msep node.
        root_msep.outlinks.append(measure_nodes[0])

        ######################################################################
        # Now, compute measure node durations from time signatures.
        #  This is slightly non-trivial. Normally, a time signature is
        #  (a) at the start of the staff, (b) right before the msep starting
        #  the measure to which it should apply. However, sometimes the msep
        #  comes up (c) at the *start* of the measure to which it should
        #  apply. We IGNORE option (c) for now.
        #
        #  - Collect all time signatures
        time_signatures = [c for c in cropobjects
                           if c.clsname in self._CONST.TIME_SIGNATURES]

        #  - Assign time signatures to measure separators that *end*
        #    the bars. (Because as opposed to the starting mseps,
        #    the end mseps are (a) always there, (b) usually on the
        #    same staff, (c) if not on the same staff, then they are
        #    an anticipation at the end of a system, and will be repeated
        #    at the beginning of the next one anyway.)
        time_signatures_to_first_measure = {}
        for t in time_signatures:
            s = self.__children(t, ['staff'])[0]
            # - Find the measure pairs
            for i, (left_msep, right_msep) in enumerate(measures):
                if s not in msep_staffs[right_msep.objid]:
                    continue
                if (left_msep is None) or (s not in msep_staffs[left_msep.objid]):
                    # Beginning of system, valid already for the current bar.
                    time_signatures_to_first_measure[t.objid] = i
                else:
                    # Use i + 1, because the time signature is valid
                    # for the *next* measure.
                    time_signatures_to_first_measure[t.objid] = i + 1

        # - Interpret time signatures.
        time_signature_durations = {t.objid: self.interpret_time_signature(t)
                                    for t in time_signatures}

        # - Reverse map: for each measure, the time signature valid
        #   for the measure.
        measure_to_time_signature = [None for _ in measures]
        time_signatures_sorted = sorted(time_signatures,
                                        key=lambda x: time_signatures_to_first_measure[x.objid])
        for t1, t2 in zip(time_signatures_sorted[:-1], time_signatures_sorted[1:]):
            affected_measures = list(range(time_signatures_to_first_measure[t1.objid],
                                      time_signatures_to_first_measure[t2.objid]))
            for i in affected_measures:
                # Check for conflicting time signatures previously
                # assigned to this measure.
                if measure_to_time_signature[i] is not None:
                    _competing_time_sig = measure_to_time_signature[i]
                    if (time_signature_durations[t1.objid] !=
                            time_signature_durations[_competing_time_sig.objid]):
                        raise ValueError('Trying to overwrite time signature to measure'
                                         ' assignment at measure {0}: new time sig'
                                         ' {1} with value {2}, previous time sig {3}'
                                         ' with value {4}'
                                         ''.format(i, t1.uid,
                                                   time_signature_durations[t1.objid],
                                                   _competing_time_sig.uid,
                                                   time_signature_durations[_competing_time_sig.objid]))

                measure_to_time_signature[i] = t1

        logging.debug('Checking that every measure has a time signature assigned.')
        for i, (msep1, msep2) in enumerate(measures):
            if measure_to_time_signature[i] is None:
                raise ValueError('Measure without time signature: {0}, between'
                                 'separators {1} and {2}'
                                 ''.format(i, msep1.uid, msep2.uid))

        # - Apply to each measure node the duration corresponding
        #   to its time signature.
        for i, m in enumerate(measure_nodes):
            _tsig = measure_to_time_signature[i]
            m.duration = time_signature_durations[_tsig.objid]

        # ...
        # Now, the "skeleton" of the precedence graph consisting
        # pf measure separator and measure nodes is complete.
        ######################################################################

        ######################################################################
        # Collecting onset-carrying objects (at this point, noteheads
        # and rests; the repeat-measure object that would normally
        # affect duration is handled through measure node durations.
        onset_objs = [c for c in cropobjects
                      if c.clsname in self._CONST.clsnames_bearing_duration]

        # Assign onset-carrying objects to measures (their left msep).
        # (This is *not* done by assigning outlinks to measure nodes,
        # we are now just factorizing the space of possible precedence
        # graphs.)
        #  - This is done by iterating over staves.
        staff_to_objs_map = collections.defaultdict(list)
        for c in onset_objs:
            ss = self.__children(c, ['staff'])
            for s in ss:
                staff_to_objs_map[s.objid].append(c)

        #  - Noteheads and rests are all connected to staves,
        #    which immediately gives us for each staff the subset
        #    of eligible symbols for each measure.
        #  - We can just take the vertical projection of each onset
        #    object and find out which measures it overlaps with.
        #    To speed this up, we can just check whether the middles
        #    of objects fall to the region delimited by the measure
        #    separators. Note that sometimes the barlines making up
        #    the measure separator are heavily bent, so it would
        #    be prudent to perhaps use just the intersection of
        #    the given barline and the current staff.

        # Preparation: we need for each valid (staff, msep) combination
        # the bounding box of their intersection, in order to deal with
        # more curved measure separators.

        msep_to_staff_projections = {}
        #: For each measure separator, for each staff it connects to,
        #  the bounding box of the measure separator's intersection with
        #  that staff.
        for msep in measure_separators:
            msep_to_staff_projections[msep.objid] = {}
            for s in msep_staffs[msep.objid]:
                intersection_bbox = self.msep_staff_overlap_bbox(msep, s)
                msep_to_staff_projections[msep.objid][s.objid] = intersection_bbox

        staff_and_measure_to_objs_map = collections.defaultdict(
                                            collections.defaultdict(list))
        #: Per staff (indexed by objid) and measure (by order no.), keeps a list of
        #  CropObjects from that staff that fall within that measure.

        # Iterate over objects left to right, shift measure if next object
        # over bound of current measure.
        ordered_objs_per_staff = {s_objid: sorted(s_objs, key=lambda x: x.left)
                                  for s_objid, s_objs in list(staff_to_objs_map.items())}
        for s_objid, objs in list(ordered_objs_per_staff.items()):
            # Vertically, we don't care -- the attachment to staff takes
            # care of that, we only need horizontal placement.
            _c_m_idx = 0   # Index of current measure
            _c_msep_right = measure_nodes[_c_m_idx].outlinks[0]
            # Left bound of current measure's right measure separator
            _c_m_right = msep_to_staff_projections[_c_msep_right.objid][s_objid][1]
            for _c_o_idx, o in objs:
                # If we are out of bounds, move to next measure
                while o.left > _c_m_right:
                    _c_m_idx += 1
                    if _c_m_idx >= len(measure_nodes):
                        raise ValueError('Object {0}: could not assign to any measure,'
                                         ' ran out of measures!'.format(o.objid))
                    _c_msep_right = measure_nodes[_c_m_idx].outlinks[0]
                    _c_m_right = msep_to_staff_projections[_c_msep_right.objid][s_objid][1]
                    staff_and_measure_to_objs_map[s_objid][_c_m_right] = []

                staff_and_measure_to_objs_map[s_objid][_c_m_right].append(o)

        # Infer precedence within the measure.
        #  - This is the difficult part.
        #  - First: check the *sum* of durations assigned to the measure
        #    against the time signature. If it fits only once, then it is
        #    a monophonic measure and we can happily read it left to right.
        #  - If the measure is polyphonic, the fun starts!
        #    With K graph nodes, how many prec. graphs are there?
        for s_objid in staff_and_measure_to_objs_map:
            for measure_idx in staff_and_measure_to_objs_map[s_objid]:
                _c_objs = staff_and_measure_to_objs_map[s_objid][measure_idx]
                measure_graph = self.measure_precedence_graph(_c_objs)

                # Connect the measure graph source nodes to their preceding
                # measure separator.
                l_msep_node = measure_nodes[measure_idx].inlinks[0]
                for source_node in measure_graph:
                    l_msep_node.outlinks.append(source_node)
                    source_node.inlinks.append(l_msep_node)

        return [root_msep]

    def measure_precedence_graph(self, cropobjects):
        """Indexed by staff objid and measure number, holds the precedence graph
        for the given measure in the given staff as a list of PrecedenceGraphNode
        objects that correspond to the source nodes of the precedence subgraph.
        These nodes then get connected to their leftwards measure separator node.

        :param cropobjects: List of CropObjects, assumed to be all from one
            measure.

        :returns: A list of PrecedenceGraphNode objects that correspond
            to the source nodes in the precedence graph for the (implied)
            measure. (In monophonic music, the list will have one element.)
            The rest of the measure precedence graph nodes is accessible
            through the sources' outlinks.

        """
        _is_monody = self.is_measure_monody(cropobjects)
        if _is_monody:
            source_nodes = self.monody_measure_precedence_graph(cropobjects)
            return source_nodes

        else:
            raise ValueError('Cannot deal with onsets in polyphonic music yet.')

    def monody_measure_precedence_graph(self, cropobjects):
        """Infers the precedence graph for a plain monodic measure.
        The resulting structure is very simple: it's just a chain
        of the onset-carrying objects from left to right."""
        nodes = []
        for c in sorted(cropobjects, key=lambda x: x.left):
            potential_durations = self.beats(c)

            # In monody, there should only be one duration
            if len(potential_durations) > 1:
                raise ValueError('Object {0}: More than one potential'
                                 ' duration, even though the measure is'
                                 ' determined to be monody.'.format(c.uid))
            duration = potential_durations[0]

            node = PrecedenceGraphNode(objid=c.objid,
                                       cropobject=c,
                                       inlinks=[],
                                       outlinks=[],
                                       duration=duration,
                                       onset=None)
            nodes.append(node)
        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            n1.outlinks.append(n2)
            n2.inlinks.append(n1)
        source_nodes = [nodes[0]]
        return source_nodes

    def is_measure_monody(self, cropobjects):
        """Checks whether the given measure is written as simple monody:
        no two of the onset-carrying objects are active simultaneously.

        Assumptions
        -----------

        * Detecting monody without looking at the time signature:
            * All stems in the same direction? --> NOPE: Violin chords in Bach...
            * All stems in horizontally overlapping noteheads in the same direction?
              --> NOPE: Again, violin chords in Bach...
            * Overlapping noteheads share a beam, but not a stem? --> this works,
              but has false negatives: overlapping quarter notes
        """
        raise NotImplementedError()

    def is_measure_chord_monody(self, cropobjects):
        """Checks whether the given measure is written as monody potentially
        with chords. That is: same as monody, but once all onset-carrying objects
        that share a stem are merged into an equivalence class."""
        raise NotImplementedError()

    def msep_staff_overlap_bbox(self, measure_separator, staff):
        """Computes the bounding box for the part of the input
        ``measure_separator`` that actually overlaps the ``staff``.
        This is implemented to deal with mseps that curve a lot,
        so that their left/right bounding box may mistakenly
        exclude some symbols from their preceding/following measure.

        Returns the (T, L, B, R) bounding box.
        """
        intersection = measure_separator.bbox_intersection(staff)
        if intersection is None:
            # Corner case: measure separator is connected to staff,
            # but its bounding box does *not* overlap the bbox
            # of the staff.
            output_bbox = staff.top, measure_separator.left, \
                          staff.bottom, measure_separator.right
        else:
            # The key step: instead of using the bounding
            # box intersection, first crop the zeros from
            # msep intersection mask (well, find out how
            # many left and right zeros there are).
            it, il, ib, ir = intersection
            msep_crop = measure_separator.mask[it, il, ib, ir]

            if msep_crop.sum() == 0:
                # Corner case: bounding box does encompass staff,
                # but there is msep foreground pixel in that area
                # (could happen e.g. with mseps only drawn *around*
                # staffs).
                output_bbox = staff.top, measure_separator.left, \
                              staff.bottom, measure_separator.right
            else:
                # The canonical case: measure separator across the staff.
                msep_crop_vproj = msep_crop.sum(axis=0)
                _dl = 0
                _dr = 0
                for i, v in enumerate(msep_crop_vproj):
                    if v != 0:
                        _dl = i
                        break
                for i in range(1, len(msep_crop_vproj)):
                    if msep_crop_vproj[-i] != 0:
                        _dr = i
                        break
                output_bbox = staff.top, measure_separator.left + _dl, \
                              staff.bottom, measure_separator.right - _dr
        return output_bbox

    def interpret_time_signature(self, time_signature,
                                 FRACTIONAL_VERTICAL_IOU_THRESHOLD=0.8):
        """Converts the time signature into the beat count (in quarter
        notes) it assigns to its following measures.

        Dealing with numeric time signatures
        ------------------------------------

        * Is there both a numerator and a denominator?
          (Is the time sig. "fractional"?)
           * If there is a letter_other child, then yes; use the letter_other
             symbol to separate time signature into numerator (top, left) and
             denominator regions.
           * If there is no letter_other child, then check if there is sufficient
             vertical separation between some groups of symbols. Given that it
             is much more likely that there will be the "fractional" structure,
             we say:

               If the minimum vertical IoU between two symbols is more than
               0.8, we consider the time signature non-fractional.

             (The threshold can be controlled through the
             FRACTIONAL_VERTICAL_IOU_THRESHOLD parameter.)

        * If yes: assign numerals to either num. (top), or denom. (bottom)
        * If not: assume the number is no. of beats. (In some scores, the
          base indicator may be attached in form of a note instead of a
          denumerator, like e.g. scores by Janacek, but we ignore this for now.
          In early music, 3 can mean "tripla", which is 3/2.)

        Dealing with non-numeric time signatures
        ----------------------------------------

        * whole-time mark is interpreted as 4/4
        * alla breve mark is interpreted as 4/4


        :returns: The denoted duration of a measure in beats.
        """
        members = sorted(self.__children(time_signature,
                                         clsnames=_CONST.TIME_SIGNATURE_MEMBERS),
                         key=lambda x: x.top)
        logging.info('Interpreting time signature {0}'.format(time_signature.uid))
        logging.info('... Members {0}'.format([m.clsname for m in members]))

        # Whole-time mark? Alla breve?
        if len(members) == 0:
            raise NotationGraphError('Time signature has no members: {0}'
                                     ''.format(time_signature.uid))

        is_whole = False
        is_alla_breve = False
        for m in members:
            if m.clsname == 'whole-time_mark':
                is_whole = True
            if m.clsname == 'alla_breve':
                is_alla_breve = True

        if is_whole or is_alla_breve:
            logging.info('Time signature {0}: whole or alla breve, returning 4.0'
                         ''.format(time_signature.uid))
            return 4.0

        # Process numerals
        logging.info('... Found numeric time signature, determining whether'
                     ' it is fractional.')

        # Does the time signature have a fraction-like format?
        is_fraction_like = True
        has_letter_other = (len([m for m in members if m.clsname == 'letter_other']) > 0)
        #  - Does it have a separator slash?
        if has_letter_other:
            logging.info('... Has fraction slash')
            is_fraction_like = True
        #  - Does it have less than 2 members?
        elif len(members) < 2:
            logging.info('... Just one member')
            is_fraction_like = False
        #  - If it has 2 or more members, determine minimal IoU and compare
        #    against FRACTIONAL_VERTICAL_IOU_THRESHOLD. If the minimal IoU
        #    is under the threshold, then consider the numerals far apart
        #    vertically so that they constitute a fraction.
        else:
            logging.info('... Must check for min. vertical overlap')
            vertical_overlaps = []
            for _i_m, m1 in enumerate(members[:-1]):
                for m2 in members[_i_m:]:
                    vertical_overlaps.append(bbox_dice(m1.bounding_box, m2.bounding_box))
            logging.info('... Vertical overlaps found: {0}'.format(vertical_overlaps))
            if min(vertical_overlaps) < FRACTIONAL_VERTICAL_IOU_THRESHOLD:
                is_fraction_like = True
            else:
                is_fraction_like = False

        numerals = sorted(self.__children(time_signature, _CONST.NUMERALS),
                          key=lambda x: x.top)
        if not is_fraction_like:
            logging.info('... Non-fractional numeric time sig.')
            # Read numeral left to right, this is the beat count
            if len(numerals) == 0:
                raise NotationGraphError('Time signature has no numerals, but is'
                                         ' not fraction-like! {0}'
                                         ''.format(time_signature.uid))
            beats = _CONST.interpret_numerals(numerals)
            logging.info('... Beats: {0}'.format(beats))
            return beats

        else:
            logging.info('... Fractional time sig.')
            # Split into numerator and denominator
            #  - Sort numerals top to bottom
            #  - Find largest gap
            #  - Everything above largest gap is numerator, everything below
            #    is denominator.
            numerals_topdown = sorted(numerals, key=lambda c: (c.top + c.bottom) / 2)
            gaps = [((c2.bottom + c2.top) / 2) - ((c1.bottom + c2.top) / 2)
                    for c1, c2 in zip(numerals_topdown[:-1], numerals_topdown[1:])]
            largest_gap_idx = max(list(range(len(gaps))), key=lambda i: gaps[i]) + 1
            numerator = numerals[:largest_gap_idx]
            denominator = numerals[largest_gap_idx:]
            beat_count = _CONST.interpret_numerals(numerator)
            beat_units = _CONST.interpret_numerals(denominator)

            beats = beat_count / (beat_units / 4)
            logging.info('...signature : {0} / {1}, beats: {2}'
                         ''.format(beat_count, beat_units, beats))

            return beats

    def onsets(self, cropobjects):
        """Infers the onsets of notes in the given cropobjects.

        The onsets are measured in beats.

        :returns: A objid --> onset dict for all notehead-type
            CropObjects.
        """
        # We first find the precedence graph. (This is the hard
        # part.)
        # The precedence graph is a DAG structure of PrecedenceGraphNode
        # objects. The infer_precedence() method returns a list
        # of the graph's source nodes (of which there is in fact
        # only one, the way it is currently defined).
        ### precedence_graph = self.infer_precedence(cropobjects)
        precedence_graph = self.infer_precedence_from_annotations(cropobjects)
        for node in precedence_graph:
            node.onset = 0

        # Once we have the precedence graph, we need to walk it.
        # It is a DAG, so we simply do a BFS from each source.
        # Whenever a node has more incoming predecessors,
        # we need to wait until they are *all* resolved,
        # and check whether they agree.
        queue = []
        # Note: the queue should be prioritized by *onset*, not number
        # of links from initial node. Leades to trouble with unprocessed
        # ancestors...
        for node in precedence_graph:
            if len(node.inlinks) == 0:
                queue.append(node)

        onsets = {}

        logging.debug('Size of initial queue: {0}'.format(len(queue)))
        logging.debug('Initial queue: {0}'.format([(q.obj.objid, q.onset) for q in queue]))

        # We will only be appending to the queue, so the
        # start of the queue is defined simply by the index.
        __qstart = 0
        __prec_clsnames = InferenceEngineConstants().clsnames_affecting_onsets
        __n_prec_nodes = len([c for c in cropobjects
                              if c.clsname in __prec_clsnames])
        __delayed_prec_nodes = dict()
        while (len(queue) - __qstart) > 0:
            # if len(queue) > 2 * __n_prec_nodes:
            #     logging.warning('Safety valve triggered: queue growing endlessly!')
            #     break

            q = queue[__qstart]
            logging.debug('Current @{0}: {1}'.format(__qstart, q.obj.uid))
            logging.debug('Will add @{0}: {1}'.format(__qstart, q.outlinks))

            __qstart += 1
            for post_q in q.outlinks:
                if post_q not in queue:
                    queue.append(post_q)

            logging.debug('Queue state: {0}'
                          ''.format([ppq.obj.objid for ppq in queue[__qstart:]]))

            logging.debug('  {0} has onset: {1}'.format(q.node_id, q.onset))
            if q.onset is not None:
                if q.onset > 0:
                    break
                onsets[q.obj.objid] = q.onset
                continue

            prec_qs = q.inlinks
            prec_onsets = [pq.onset for pq in prec_qs]
            # If the node did not yet get all its ancestors processed,
            # send it down the queue.
            if None in prec_onsets:
                logging.warning('Found node with predecessor that has no onset yet; delaying processing: {0}'
                                ''.format(q.obj.uid))
                queue.append(q)
                if q in __delayed_prec_nodes:
                    logging.warning('This node has already been delayed once! Breaking.')
                    logging.warning('Queue state: {0}'
                                    ''.format([ppq.obj.objid for ppq in queue[__qstart:]]))
                    break
                else:
                    __delayed_prec_nodes[q.obj.objid] = q
                    continue

            prec_durations = [pq.duration for pq in prec_qs]

            logging.debug('    Prec_onsets @{0}: {1}'.format(__qstart - 1, prec_onsets))
            logging.debug('    Prec_durations @{0}: {1}'.format(__qstart - 1, prec_durations))

            onset_proposals = [o + d for o, d in zip(prec_onsets, prec_durations)]
            if min(onset_proposals) != max(onset_proposals):
                if self.strategy.permissive_desynchronization:
                    logging.warning('Object {0}: onsets not synchronized from'
                                    ' predecessors: {1}'.format(q.obj.uid,
                                                                onset_proposals))
                    onset = max(onset_proposals)
                else:
                    raise ValueError('Object {0}: onsets not synchronized from'
                                     ' predecessors: {1}'.format(q.obj.uid,
                                                                 onset_proposals))
            else:
                onset = onset_proposals[0]

            q.onset = onset
            # Some nodes do not have a Node assigned.
            if q.obj is not None:
                onsets[q.obj.objid] = onset
                ### DEBUG -- add this to the Data dict
                q.obj.data['onset_beats'] = onset

        return onsets

    def __children(self, c, clsnames):
        """Retrieve the children of the given Node ``c``
        that have class in ``clsnames``."""
        return [self._cdict[o] for o in c.outlinks
                if self._cdict[o].clsname in clsnames]

    def __parents(self, c, clsnames):
        """Retrieve the parents of the given Node ``c``
        that have class in ``clsnames``."""
        return [self._cdict[i] for i in c.inlinks
                if self._cdict[i].clsname in clsnames]

    def __warning_or_error(self, message):
        if self.strategy.permissive:
            logging.warn(message)
        else:
            raise ValueError(message)

    def process_ties(self, cropobjects, durations, onsets):
        """Modifies the durations and onsets so that ties are taken into
        account.

        Every left-hand note in a tie gets its duration extended by the
        right-hand note's duration. Every right-hand note's onset is removed.

        :returns: the modified durations and onsets.
        """
        logging.info('Processing ties...')
        g = NotationGraph(cropobjects=cropobjects)

        def _get_tie_notes(_tie, graph):
            notes = graph.parents(_tie, classes=['notehead-full', 'notehead-empty'])
            if len(notes) == 0:
                raise NotationGraphError('No notes from tie {0}'.format(_tie.uid))
            if len(notes) == 1:
                return [notes[0]]
            if len(notes) > 2:
                raise NotationGraphError('More than two notes from tie {0}'.format(_tie.uid))
            # Now it has to be 2
            l, r = sorted(notes, key=lambda n: n.left)
            return l, r

        def _is_note_left(c, _tie, graph):
            tie_notes = _get_tie_notes(_tie, graph)
            if len(tie_notes) == 2:
                l, r = tie_notes
                return l.objid == c.objid
            else:
                return True

        new_onsets = copy.deepcopy(onsets)
        new_durations = copy.deepcopy(durations)
        # Sorting notes right to left. This means: for notes in the middle
        # of two ties, its duration is already updated and it can be removed from
        # the new onsets dict by the time we process the note on the left
        # of the leftward tie (its predecessor).
        for k in sorted(onsets, key=lambda x: onsets[x], reverse=True):
            ties = g.children(k, classes=['tie'])
            if len(ties) == 0:
                continue

            if len(ties) > 1:
                # Pick the rightmost tie (we're processing onsets from the right)
                tie = max(ties, key=lambda x: x.left)
            else:
                tie = ties[0]
            n = g[k]
            tie_notes = _get_tie_notes(tie, graph=g)
            if len(tie_notes) != 2:
                continue

            l, r = tie_notes
            if l.objid == n.objid:
                logging.info('Note {0} is left in tie {1}'
                             ''.format(l.uid, tie.uid))
                new_durations[l.objid] += new_durations[r.objid]
                del new_onsets[r.objid]

        new_durations = {k: new_durations[k] for k in new_onsets}
        return new_durations, new_onsets


class PrecedenceGraphNode(object):
    """A helper plain-old-data class for onset extraction.
    The ``inlinks`` and ``outlinks`` attributes are lists
    of other ``PrecedenceGraphNode`` instances.
    """
    def __init__(self, objid=None, cropobject=None, inlinks=None, outlinks=None,
                 onset=None, duration=0):
        # Optional link to CropObjects, or just a placeholder ID.
        self.obj = cropobject
        if objid is None and cropobject is not None:
            objid = cropobject.objid
        self.node_id = objid

        self.inlinks = []
        if inlinks:
            self.inlinks = inlinks
        self.outlinks = []
        if outlinks:
            self.outlinks = outlinks

        self.onset = onset
        '''Counting from the start of the musical sequence, how many
        beat units pass before this object?'''

        self.duration = duration
        '''By how much musical time does the object delay the onsets
        of its descendants in the precedence graph?'''


class MIDIBuilder(object):

    def midi_matrix_to_pdo(self, midi_matrix, framerate=20, tempo=120):
        """Builds the pitch, duration and onset dicts from a given MIDI
        matrix. Does *not* take into account possible re-articulations:
        repeated adjacent notes are transformed into just one.

        :param midi_matrix: A ``128 x n_frames`` binary numpy array.
            Expected to be in the less intuitive format, where pitch
            ``J`` is encoded in row ``(128 - J)`` -- you would plot this
            with ``origin=lower`` in the ``imshow()`` call.

        :param FPS: each frame in the MIDI matrix corresponds to ``1 / FPS``
            of a second. Used together with ``tempo`` to determine durations
            in beats.

        :param tempo: The tempo in which the MIDI matrix should be interpreted.
            This does not actually matter for the output MIDI -- you can
            balance it out by using a different ``FPS`` value. However, it is
            necessary to compute durations and onsets in beats, since this is
            what the MIDI building functions in ``midiutil.MidiFile`` expect.

        :returns: ``pitches, durations, onsets``. These are dicts indexed
            by note ID (equivalent to notehead objids in MuNG context).
            Pitches contain for each note the MIDI pitch code, durations
            contain its duration in beats, and onsets contain its onset
            in beats.
        """
        pitches = dict()
        durations = dict()
        onsets = dict()

        # Collect pitch activities.

        # For each pitch, contains a list of (start_frame, end_frame+1) pairs.
        activities = collections.defaultdict(list)

        n_frames = midi_matrix.shape[1]
        n_pitch_classes = midi_matrix.shape[0]
        currently_active = dict()
        for i_frame in range(n_frames):
            # Collect onsets
            frame = midi_matrix[:, i_frame]
            for idx, entry in enumerate(frame):
                # pitch = n_pitch_classes - idx
                pitch = idx
                if entry != 0:
                    if pitch not in currently_active:
                        # Start activity
                        currently_active[pitch] = i_frame
                else:
                    if pitch in currently_active:
                        activities[pitch].append((currently_active[pitch], i_frame))
                        del currently_active[pitch]

        # Convert pitch activities into onset/pitch/duration dicts.
        notes = []
        for pitch in activities:
            for onset_frame, end_frame in activities[pitch]:
                notes.append((onset_frame, pitch, end_frame - onset_frame))
        # Sort by start and from lowest to highest:
        ordered_by_start_frame = sorted(notes)

        # Distribute into pitch/duration/onset dicts
        for event_idx, (onset_frame, pitch, duration_frames) in enumerate(notes):
            onset_beats = self.frames2beats(onset_frame,
                                            framerate=framerate,
                                            tempo=tempo)
            duration_beats = self.frames2beats(duration_frames,
                                               framerate=framerate,
                                               tempo=tempo)
            pitches[event_idx] = pitch
            durations[event_idx] = duration_beats
            onsets[event_idx] = onset_beats

        logging.debug('{} note events, last onset: beat {} (seconds: {})'
                      ''.format(len(notes), notes[-1][0], notes[-1][0] * tempo / 60.))

        return pitches, durations, onsets

    def frames2beats(self, n_frames, framerate, tempo):
        """Converts a number of frames to duration in beats,
        given a framerate and tempo."""
        return (n_frames / float(framerate)) * (tempo / 60.)

    def build_midi(self, pitches, durations, onsets, selection=None, tempo=120):
        from midiutil.MidiFile import MIDIFile

        # create your MIDI object
        mf = MIDIFile(1)     # only 1 track
        track = 0   # the only track

        time = 0    # start at the beginning
        mf.addTrackName(track, time, "Sample Track")
        mf.addTempo(track, time, tempo)

        channel = 0
        volume = 100

        keys = list(pitches.keys())

        min_onset = 0
        if selection is not None:
            keys = [k for k in keys if k in selection]
            min_onset = min([onsets[k] for k in keys if k in onsets])

        for objid in keys:
            if (objid in onsets) and (objid in durations):
                pitch = pitches[objid]
                onset = onsets[objid] - min_onset
                duration = durations[objid]
                mf.addNote(track, channel, pitch, onset, duration, volume)

        return mf


def play_midi(midi,
              tmp_dir,
              soundfont='~/.fluidsynth/FluidR3_GM.sf2',
              cleanup=False):
    """Plays (or attempts to play) the given MIDIFile object.

    :param midi: A ``midiutils.MidiFile.MIDIFile`` object
        containing the data that you wish to play.

    :param tmp_dir: A writeable directory where the MIDI will be
        exported into a temporary file.

    :param soundfont: A *.sf2 soundfont for FluidSynth to load.
    """
    from midi2audio import FluidSynth

    import uuid
    tmp_midi_path = os.path.join(tmp_dir, 'play_' + str(uuid.uuid4())[:8] + '.mid')
    with open(tmp_midi_path, 'wb') as hdl:
        midi.writeFile(hdl)
    if not os.path.isfile(tmp_midi_path):
        logging.warn('Could not write MIDI data to temp file {0}!'.format(tmp_midi_path))
        return

    fs = FluidSynth(soundfont)
    fs.play_midi(tmp_midi_path)
    # Here's hoping it's a blocking call. Otherwise, just leave the MIDI;
    # MUSCIMarker cleans its tmp dir whenever it exits.
    if cleanup:
        os.unlink(tmp_midi_path)


def align_mung_with_midi(cropobjects, midi_file):
    """Aligns cropobjects (noteheads) with MIDI note_on
    events. Outputs a list of pairs ``(c, m)``.

    Should be agnostic w.r.t. the chosen MIDI representation?

    Assumes all noteheads have corresponding pitch, onset & duration
    information. The alignment is basically finding how beats
    translate to time in the MIDI file. For a "dead" rendered
    MIDI, this is easy, since (assuming a constant tempo) there
    is a constant ratio. For a MIDI performance, this may be much
    more difficult.

    Assuming the events are at least in the correct
    order, we can use Dynamic Time Warping, using only pitch matching
    as the cost function as the simplest baseline.

    The ``m`` element of each returned item is then the onset time
    in the MIDI, in ticks. This way, the alignment algorithm is fully
    independent of the representation one chooses for the MIDI data,
    while extracting the relevant information.

    For real performances with errors, one would have to use pitch *distance*
    as the cost function, and incorprate the time differences into
    the cost function in order to decide which events to align
    to noteheads even though they may be errors, and which to leave
    unaligned. Then, we would also have to output the full info of
    the corresponding MIDI event.
    """
    # Read MIDI.
    from midi import read_midifile
    midi = read_midifile(midi_file)

    # Alignment algorithm:
    # - find in MIDI events list the closest corresponding
    #   object for each notehead.

    raise NotImplementedError()