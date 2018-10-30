"""This module acts as an abstraction over the dataset.

It mostly implements utility functions, like getting the absolute
path to a specific image in the CVC-MUSCIMA dataset, specified
by the writer, number, distortion, and mode.

Environmental variables
-----------------------

* ``CVC_MUSCIMA_ROOT``
* ``MUSCIMA_PLUSPLUS_ROOT``

The dataset root environmental variables are used as default roots
for retrieving the dataset files. If they are not set, you will
have to supply the roots to the respective functions that manipulate
these layers of MUSCIMA++.
"""
from __future__ import print_function, unicode_literals, division

from builtins import str
from builtins import range
from builtins import object
import logging
import os

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


def _get_cvc_muscima_root():
    if 'CVC_MUSCIMA_ROOT' in os.environ:
        CVC_MUSCIMA_ROOT = os.environ['CVC_MUSCIMA_ROOT']
        return CVC_MUSCIMA_ROOT
    else:
        logging.info('muscima.dataset: environmental variable CVC_MUSCIMA_ROOT not defined.')
        return None

CVC_MUSCIMA_ROOT = _get_cvc_muscima_root()

##############################################################################


def _get_mff_muscima_root():
    if 'MUSCIMA_PLUSPLUS_ROOT' in os.environ:
        MUSCIMA_PLUSPLUS_ROOT = os.environ['MUSCIMA_PLUSPLUS_ROOT']
        return MUSCIMA_PLUSPLUS_ROOT
    else:
        logging.info('muscima.dataset: environmental variable MUSCIMA_PLUSPLUS_ROOT not defined.')
        return None

MUSCIMA_PLUSPLUS_ROOT = _get_mff_muscima_root()

##############################################################################


class CVC_MUSCIMA(object):
    """The :class:`CVC_MUSCIMA` class implements a wrapper around
    the CVC-MUSCIMA dataset file structure that allows easy retrieval
    of filenames based on the page number (1 - 20), writer number
    (1 - 50), distortion, and mode (full image, staffline pixels only, or
    non-staffline pixels only).

    This functionality is defined in :meth:`imfile`.
    """

    DISTORTIONS = [
        'curvature',
        'ideal',
        'interrupted',
        'kanungo',
        'rotated',
        'staffline-thickness-variation-v1',
        'staffline-thickness-variation-v2',
        'staffline-y-variation-v1',
        'staffline-y-variation-v2',
        'thickness-ratio',
        'typeset-emulation',
        'whitespeckles',
    ]

    MODES = ['full', 'symbol', 'staff_only']

    def __init__(self, root=_get_cvc_muscima_root(), validate=False):
        """The dataset is instantiated by providing the path to its root
        directory. If the ``CVC_MUSCIMA_ROOT`` variable is set, you do not
        have to provide anything."""
        if root is None:
            raise ValueError('CVC_MUSCIMA class needs the dataset root to serve'
                             ' any useful purpose: either set the CVC_MUSCIMA_ROOT'
                             ' environmental variable, or supply the root manually.')
        if not os.path.isdir(root):
            logging.warning('Instantiating CVC-MUSCIMA dataset wrapper without'
                            ' a valid root: the path {0} does not lead to'
                            ' a directory.'.format(root))

        if validate and not self.validate(fail_early=True):
            raise ValueError('CVC_MUSCIMA innstance with root {0} does not'
                             ' represent a valid CVC-MUSCIMA dataset copy.'
                             ''.format(self.root))

        self.root = root

    def imfile(self, page, writer, distortion='ideal', mode='full'):
        """Construct the path leading to the file of the CVC-MUSCIMA image
        with the specified page (1 - 20), writer (1 - 50), distortion
        (see ``CVC_MUSCIMA_DISTORTIONS``), and mode (``full``, ``symbol``,
        ``staff_only``).

        This is the primary interface that the CVC_MUSCIMA class provides.
        """
        if distortion not in self.DISTORTIONS:
            raise ValueError('Distortion {0} not available in CVC-MUSCIMA.'
                             ' Choose one of:\n{1}'
                             ''.format(distortion, self.DISTORTIONS))
        if mode not in self.MODES:
            raise ValueError('Image mode {0} not available in CVC-MUSCIMA.'
                             ' Choose one of:\n{1}'
                             ''.format(mode, self.MODES))

        fname = os.path.join(self.root, distortion,
                             self._number2writer_dir(writer),
                             self._mode2dir(mode),
                             self._number2page_file(page))
        if not os.path.isfile(fname):
            logging.warning('The requested file {0} should be available,'
                            ' but does not seem to be there. Are you sure'
                            ' the CVC-MUSCIMA root is set correctly? ({1})'
                            ''.format(fname, self.root))
        return fname

    def _number2page_file(self, n):
        if (n < 1) or (n > 20):
            raise ValueError('Invalid CVC-MUSCIMA score number {0}.'
                             ' Valid only between 1 and 20.'.format(n))
        if n < 10:
            return 'p0' + '0' + str(n) + '.png'
        else:
            return 'p0' + str(n) + '.png'

    def _number2writer_dir(self, n):
        if (n < 1) or (n > 50):
            raise ValueError('Invalid MUSCIMA writer number {0}.'
                             ' Valid only between 1 and 50.'.format(n))
        if n < 10:
            return 'w-0' + str(n)
        else:
            return 'w-' + str(n)

    def _mode2dir(self, mode):
        if mode == 'full':
            return 'image'
        elif mode == 'symbol':
            return 'symbol'
        elif mode == 'staff_only':
            return 'gt'

    def validate(self, fail_early=True):
        """Checks whether the instantiated CVC_MUSCIMA instance really
        corresponds to the CVC-MUSCIMA dataset: all the 12 x 1000 expected
        CVC-MUSCIMA files should be present.

        :param fail_early: If ``True``, will return as soon as it encounters
            a missing file, if ``False``, will keep going through all the files
            and find out which ones are missing. (Default: ``True``)

        :returns: ``True`` if the dataset is OK, ``False`` if any file
            is missing.
        """
        _missing = []
        for d in self.DISTORTIONS:
            for m in self.MODES:
                for p in range(1, 21):
                    for w in range(1, 51):
                        f = self.imfile(page=p, writer=w, distortion=d, mode=m)
                        if not os.path.isfile(f):
                            if fail_early:
                                logging.warning('Missing file: {0}'.format(f))
                                return False
                            _missing.append(f)
        if len(_missing) > 0:
            logging.warning('Missing files in CVC_MUSCIMA instance with root {0}:'
                            '\n{1}'.format(self.root, '\n'.join(_missing)))
            return False
        return True
