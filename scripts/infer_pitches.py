#!/usr/bin/env python
"""This is a script that takes the full-grown notation graph
and recovers for each notehead the pitch to which it corresponds.

Assumptions
-----------

* Clefs are used in a standard way: G-clef on 4th, C-clef on 3rd, F-clef
  on 2nd staffline.
* Key signatures are used in a standard way, so that we can rely on counting
  the accidentals.
* Accidentals are valid up until the end of the bar.

We are currently NOT processing any transpositions.

Representation
--------------

Notes are not noteheads. Pitch is associated with a note, and it is derived
from the notehead's subgraph. The current goal of this exercise is obtaining
MIDI, so we discard in effect information about what is e.g. a G-sharp
and A-flat.

"""
import argparse
import logging
import os
import time

from mung.inference.inference import PitchInferenceEngine, OnsetsInferenceEngine,  MIDIBuilder
from mung.io import read_nodes_from_file, export_node_list


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-a', '--annot', action='store', required=True,
                        help='The annotation file for which the staffline and staff'
                             ' CropObject relationships should be added.')
    parser.add_argument('-e', '--export', action='store',
                        help='A filename to which the output CropObjectList'
                             ' should be saved. If not given, will print to'
                             ' stdout.')
    parser.add_argument('-m', '--midi', action='store',
                        help='A filename to which to export the MIDI file'
                             ' for the given score.')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')
    return parser


##############################################################################


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    # Your code goes here
    if not os.path.isfile(args.annot):
        raise ValueError('Annotation file {0} not found!'
                         ''.format(args.annot))
    cropobjects = read_nodes_from_file(args.annot)

    pitch_inference_engine = PitchInferenceEngine()
    time_inference_engine = OnsetsInferenceEngine(nodes=cropobjects)

    logging.info('Running pitch inference.')
    pitches, pitch_names = pitch_inference_engine.infer_pitches(cropobjects,
                                                                with_names=True)

    # Export
    logging.info('Adding pitch information to <Data> attributes.')
    for c in cropobjects:
        if c.objid in pitches:
            midi_pitch_code = pitches[c.objid]
            pitch_step, pitch_octave = pitch_names[c.objid]
            if c.data is None:
                c.data = dict()
            c.data['midi_pitch_code'] = midi_pitch_code
            c.data['normalized_pitch_step'] = pitch_step
            c.data['pitch_octave'] = pitch_octave

    logging.info('Adding duration info to <Data> attributes.')
    durations = time_inference_engine.durations(cropobjects)
    logging.info('Total durations: {0}'.format(len(durations)))
    for c in cropobjects:
        if c.objid in durations:
            c.data['duration_beats'] = durations[c.objid]

    logging.info('Some durations: {0}'.format(sorted(durations.items())[:10]))

    logging.info('Adding onset info to <Data> attributes.')
    onsets = time_inference_engine.onsets(cropobjects)
    logging.info('Total onsets: {0}'.format(len(onsets)))
    for c in cropobjects:
        if c.objid in onsets:
            c.data['onset_beats'] = onsets[c.objid]

    if args.export is not None:
        with open(args.export, 'w') as hdl:
            hdl.write(export_node_list(cropobjects))
            hdl.write('\n')
    else:
        print(export_node_list(cropobjects))

    if args.midi is not None:
        midi_builder = MIDIBuilder()
        mf = midi_builder.build_midi(pitches, durations, onsets)
        with open(args.midi, 'wb') as hdl:
            mf.writeFile(hdl)

    _end_time = time.clock()
    logging.info('infer_pitches.py done in {0:.3f} s'.format(_end_time - _start_time))


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
