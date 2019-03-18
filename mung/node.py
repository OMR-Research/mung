# -*- coding: utf-8 -*-
"""This module implements a Python representation of the Node,
the basic unit of annotation. See the :class:`Node` documentation."""
from __future__ import print_function, unicode_literals, division

from builtins import zip
from builtins import map
from builtins import str
from builtins import range
from builtins import object
import copy
import itertools
import logging
from typing import List, Any

import numpy

from mung.utils import compute_connected_components

__version__ = "1.0"
__author__ = "Jan Hajic jr."

CROPOBJECT_MASK_ORDER = 'C'


#: The Node mask uses this numpy ordering when flattening the data.

##############################################################################


class Node(object):
    """One annotated object.

    The Node represents one instance of an annotation. It implements
    the following attributes:

    * ``node_id``: the unique number of the given annotation instance in the set
      of annotations encoded in the containing `CropObjectList`.
    * ``unique_id``: the global unique identifier of the annotation instance. String.
      See :meth:`Node.parse_unique_id` method for format details.
    * ``class_name``: the name of the label that was given to the annotation
      (this is the human-readable string such as ``notehead-full``).
    * ``top``: the vertical dimension (row) of the upper left corner pixel.
    * ``left``: the horizontal dimension (column) of the upper left corner pixel.
    * ``bottom``: the vertical dimension (row) of the lower right corner pixel + 1,
      so that you can index the corresponding image rows using
      ``img[c.top:c.bottom]``.
    * ``right``: the horizontal dimension (row) of the lower right corner pixel + 1,
      so that you can index the corresponding image columns using
      ``img[:, c.left:c.right]``.
    * ``width``: the amount of rows that the Node spans.
    * ``height``: the amount of columns that the Node spans.
    * ``mask``: a binary (0/1) numpy array that denotes the area within the
      Node's bounding box (specified by ``top``, ``left``, ``height``
      and ``width``) that the Node actually occupies. If the mask is
      ``None``, the object is understood to occupy the entire bounding box.
    * ``data``: a dictionary that can be empty, or can contain anything. It is
      generated from the optional ``<Data>`` element of a Node.

    Constructing a simple Node that consists of the "b"-like flat music
    notation symbol (never mind the ``unique_id`` for now):

    >>> top = 10
    >>> left = 15
    >>> height = 10
    >>> width = 4
    >>> mask = numpy.array([[1, 1, 0, 0],
    ...                     [1, 0, 0, 0],
    ...                     [1, 0, 0, 0],
    ...                     [1, 0, 0, 0],
    ...                     [1, 0, 1, 1],
    ...                     [1, 1, 1, 1],
    ...                     [1, 0, 0, 1],
    ...                     [1, 0, 1, 1],
    ...                     [1, 1, 1, 0],
    ...                     [0, 1, 0, 0]])
    >>> class_name = 'flat'
    >>> unique_id = 'MUSCIMA++_1.0___mung.node.Node.doctest___0'
    >>> node = Node(node_id=0, class_name=class_name,
    ...                top=top, left=left, height=height, width=width,
    ...                inlinks=[], outlinks=[],
    ...                mask=mask,
    ...                unique_id=unique_id)

    CropObjects can also form graphs, using the following attributes:

    * ``outlinks``: Outgoing edges. A list of integers; it is assumed they are
      valid ``node_id`` within the same global/doc namespace.
    * ``inlinks``: Incoming edges. A list of integers; it is assumed they are
      valid ``node_id`` within the same global/doc namespace.

    So far, Node graphs do not support multiple relationship types.

    **Unique identification**

    The ``unique_id`` serves to identify the Node uniquely,
    at least within the MUSCIMA dataset system. (We anticipate further
    versions of the dataset, and need to plan for that.)

    To uniquely identify a Node, we need three "levels":

    * The "global", **dataset-level identification**: which dataset is this
      Node coming from? (For this dataset: ``MUSCIMA++_1.0``)
    * The "local", **document-level identification**: which document
      (within the given dataset) is this Node coming from?
      For MUSCIMA++ 1.0, this will usually be a string like
      ``CVC-MUSCIMA_W-35_N-08_D-ideal``, derived from the filename
      under which the CropObjectList containing the given Node
      is stored.
    * The **within-document identification**, which is identical
      to the ``node_id``.

    These three components are joined together into one string by
    a delimiter: ``___``

    The full ``unique_id`` of a Node then might look like this::

      MUSCIMA-pp_1.0___CVC-MUSCIMA_W-35_N-08_D-ideal___611

    You will need to use UIDs whenever you are combining CropObjects
    from different documents, and/or datasets. (If you are really combining
    datasets, make sure you know what you are doing -- some annotation
    instructions may change between versions, so objects of the same class
    might not exactly correspond to each other...) The dataset and document
    names are available through appropriate instance attributes:

    >>> node.doc
    'mung.node.Node.doctest'
    >>> node.dataset
    'MUSCIMA++_1.0'

    If you supply no ``unique_id`` at initialization time, a default UID will
    be used:

    >>> node.default_unique_id
    'MUSCIMA_DEFAULT_DATASET_PLACEHOLDER___default-document___0'

    (Don't abuse the default, though! It's intended just for transitioning
    documents without UIDs to those that have them.)

    On the other hand, the ``node_id`` is a field intended to uniquely identify
    a Node within the scope of one Node list (one annotation
    document).

    .. caution::

        The scope of unique identification within MUSCIMA++ is only within
        a ``<CropObjectList>``. Don't use ``node_id`` to mix CropObjects from
        multiple files!

    **CropObjects and images**

    CropObjects and images are not tightly bound. This is because the same
    object can apply to multiple images: in the case of the CVC-MUSCIMA dataset,
    for example, the same CropObjects are present both in the full image
    and in the staff-less image. The limitation here is that CropObjects
    are based on exact pixels, so in order to retain validity, the images
    must correspond to each other exactly, as "layers".

    Because CropObjects do not correspond to any given image, there is
    no facility in the data format to link them to a specific one. You have to
    take care of matching Node annotations to the right images by yourself.

    The ``Node`` class implements some interactions with images.

    To recover the area corresponding to a Node `c`, use:

    >>> if node.mask is not None: crop = img[node.top:node.bottom, node.left:node.right] * node.mask  #doctest: +SKIP
    >>> if node.mask is None: crop = img[node.top:node.bottom, node.left:node.right]               #doctest: +SKIP

    Because this is clunky, we have implemented the following to get the crop:

    >>> crop = node.project_to(img)    #doctest: +SKIP

    And to get the Node projected onto the entire image:

    >>> crop = node.project_on(img)    #doctest: +SKIP

    Above, note the multiplicative role of the mask: while we typically would
    expect the mask to be binary, in principle, this is not strictly necessary.
    You could supply a different mask interpration, such as probabilistic.
    However, we strongly advise not to misuse this feature unless you have
    a really good reason; remember that the Node is supposed to represent
    an annotation of a given image. (One possible use for a non-binary mask
    that we can envision is aggregating multiple annotations of the same
    image.)

    For visualization, there is a more sophisticated method that renders
    the Node as a transparent colored transparent rectangle over
    an RGB image. (NOTE: this really changes the input image!)

    >>> c_obj.render(img)           #doctest: +SKIP
    >>> plt.imshow(img); plt.show() #doctest: +SKIP

    However, `Node.render()` currently does not support rendering
    the mask.

    **Disambiguating class names**

    Since the class names are present
    through the ``class_name`` attribute (``<MLClassName>`` element),
    matching the list is no longer necessary for general understanding
    of the file. The MLClassList file serves as a disambiguation tool:
    there may be multiple annotation projects that use the same names
    but maybe define them differently and use different guidelines,
    and their respective MLClassLists allow you to interpret the symbol
    names correctly, in light of the corresponding set of definitions.

    .. note::

        In MUSCIMarker, the MLClassList is currently necessary to define
        how CropObjects are displayed: their color. (All noteheads are red,
        all barlines are green, etc.) The other function, matching names
        to ``clsid``, has been superseeded by the ``class_name`` Node
        attribute.

    **Merging CropObjects**

    To merge a list of CropObjects into a new one, you need to:

    * Compute the new object's bounding box: ``croobjects_merge_bbox()``
    * Compute the new object's mask: ``cropobjects_merge_mask()``
    * Determine the clsid and node_id of the new object.

    Since node_id and clsid of merges may depend on external settings
    and generally cannot be reliably determined from the merged
    objects themselves (e.g. the merge of a notehead and a stem
    should be a new note symbol), you need to supply them externally.
    However, the bounding box and mask can be determined. The bounding
    box is computed simply as the smallest bounding box that
    encompasses all the CropObjects, and the mask is an OR operation
    over the individual masks (or None, if the CropObjects don't
    have masks). Note that the merge cannot deal with a situation
    where only some of the objects have a mask.

    **Implementation notes on the mask**

    The mask is a numpy array that will be saved using run-length encoding.
    The numpy array is first flattened, then runs of successive 0's and 1's
    are encoded as e.g. ``0:10`` for a run of 10 zeros.

    How much space does this take?

    Objects tend to be relatively convex, so after flattening, we can expect
    more or less two runs per row (flattening is done in ``C`` order). Because
    each run takes (approximately) 5 characters, each mask takes roughly ``5 * n_rows``
    bytes to encode. This makes it efficient for objects wider than 5 pixels, with
    a compression ratio approximately ``n_cols / 5``.
    (Also, the numpy array needs to be made C-contiguous for that, which
    explains the ``order='C'`` hack in ``set_mask()``.)
    """

    def __init__(self, node_id, class_name, top, left, width, height,
                 outlinks=None, inlinks=None,
                 mask=None,
                 unique_id=None,
                 data=None):
        # type: (int, str, int, int, int, int, List[int], List[int], numpy.ndarray, str, Any) -> Node
        self.node_id = node_id

        self.class_name = class_name
        self.x = top
        self.y = left
        self.width = width
        self.height = height

        self.to_integer_bounds()

        # The mask presupposes integer bounds.
        # Applied relative to Node bounds, not the whole image.
        self.mask = None
        self.set_mask(mask)

        if inlinks is None:
            inlinks = []
        self.inlinks = inlinks

        if outlinks is None:
            outlinks = []
        self.outlinks = outlinks

        if unique_id is None:
            unique_id = self.default_unique_id
        self.set_unique_id(unique_id)

        self.is_selected = False

        if data is None:
            data = dict()
        self.data = data

    ##########################################################################
    # Dealing with unique identification of a Node, also across
    # anticipated dataset versions.

    UID_DELIMITER = '___'
    #: Delimits the Node UID fields (global, document namespaces, node_id)

    UID_DEFAULT_DATASET_NAMESPACE = 'MUSCIMA_DEFAULT_DATASET_PLACEHOLDER'
    #: Default dataset name for CropObjects.

    UID_DEFAULT_DOCUMENT_NAMESPACE = 'default-document'

    #: Default document name for CropObjects.

    @property
    def default_unique_id(self):
        # type: () -> str
        """Constructs the default ``unique_id`` that the Node would
        have, unless one was supplied at initialization.

        >>> node.default_unique_id   # doctest: +SKIP
        'MUSCIMA_DEFAULT_DATASET_PLACEHOLDER___default-document___0'
        """
        return self.UID_DELIMITER.join([self.UID_DEFAULT_DATASET_NAMESPACE,
                                        self.UID_DEFAULT_DOCUMENT_NAMESPACE,
                                        str(self.node_id)])

    def parse_unique_id(self):
        # type: () -> (str, str, int)
        """Parse the unique identifier of the Node. This
        breaks down the UID into the global namespace, document
        namespace (ie. CropObjectList name -- usually per image),
        and the numeric ID of the Node within one CropObjectList.
        This numeric ID should always match the ``node_id``, which
        acts as the "technical" identifier, since it is known to be
        an integer and therefore usable for e.g. indexing within
        the MUSCIMarker annotation app.

        See :meth:`_parse_uid` for format & test. Compared
        to :meth:`_parse_uid`, this method checks the parsed ``node_id``
        in the ``unique_id`` against this Node's ``node_id``,
        to verify that the UID is really valid for this object.

        The delimiter is expected to be ``___``
        (kept as ``Node.UID_DELIMITER``)
        """
        global_namespace, document_namespace, node_id = self._parse_uid(self.unique_id)
        # Dealing with missing unique_id
        if node_id is None:
            node_id = self.node_id

        if node_id != self.node_id:
            raise ValueError('Got Node with different numeric ID'
                             ' in UID and technical node_id. UID record:'
                             ' {0}, node_id: {1}'.format(node_id, self.node_id))
        return global_namespace, document_namespace, node_id

    @staticmethod
    def _parse_uid(uid):
        # type: (str) -> (str,str,int)
        """Parse the unique identifier of the Node. This
        breaks down the UID into the global namespace, document
        namespace (ie. CropObjectList name -- usually per image),
        and the numeric ID of the Node within one CropObjectList.

        The delimiter is expected to be ``___``
        (kept as ``Node.UID_DELIMITER``)

        >>> Node._parse_uid('MUSCIMA++_1.0___CVC-MUSCIMA_W-05_N-19_D-ideal___424')
        ('MUSCIMA++_1.0', 'CVC-MUSCIMA_W-05_N-19_D-ideal', 424)

        :returns: ``global_namespace, document_namespace, node_id`` triplet.
            The namespaces are strings, ``node_id`` is an integer. If ``unique_id``
            is ``None``, returns ``None`` as ``node_id`` and expects it
            to be filled in from the caller Node instance.
        """
        if uid is None:
            global_namespace = Node.UID_DEFAULT_DATASET_NAMESPACE
            document_namespace = Node.UID_DEFAULT_DOCUMENT_NAMESPACE
            node_id = None
        else:
            global_namespace, document_namespace, node_id_string = uid.split(Node.UID_DELIMITER)
            node_id = int(node_id_string)
        return global_namespace, document_namespace, node_id

    @staticmethod
    def build_unique_id(global_namespace, document_namespace, node_id):
        return Node.UID_DELIMITER.join([str(global_namespace),
                                        str(document_namespace),
                                        str(node_id)])

    def set_unique_id(self, unique_id):
        """Assigns the given ``unique_id`` to the Node. This is the way
        to do it, do not assign directly to ``cropobject.unique_id``! You need
        to update other things (and perform integrity checks) when changing
        the unique ID! See :class:`Node` class documentation for
        information on how ``unique_id`` attributes work.

        Do **NOT** use this function, unless you know what you are doing!
        You could mess up the integrity of your copy of the dataset, and
        you'd have to download it again...
        """
        self.unique_id = unique_id
        self._dataset_namespace, self._document_namespace, self._instance = \
            self.parse_unique_id()

    def set_doc(self, docname):
        new_uid = self.UID_DELIMITER.join([self._dataset_namespace,
                                           docname,
                                           str(self._instance)])
        self.set_unique_id(new_uid)

    def set_dataset(self, dataset_name):
        new_uid = self.UID_DELIMITER.join([dataset_name,
                                           self._document_namespace,
                                           str(self._instance)])
        self.set_unique_id(new_uid)

    def set_mask(self, mask):
        """Sets the Node's mask to the given array. Performs
        some compatibilty checks: size, dtype (converts to ``uint8``)."""
        if mask is None:
            self.mask = None
        else:
            # Check dimension
            t, l, b, r = self.bbox_to_integer_bounds(self.top,
                                                     self.left,
                                                     self.bottom,
                                                     self.right)  # .count()
            if mask.shape != (b - t, r - l):
                raise ValueError('Mask shape {0} does not correspond'
                                 ' to integer shape {1} of Node.'
                                 ''.format(mask.shape, (b - t, r - l)))
            if str(mask.dtype) != 'uint8':
                logging.debug('Node.set_mask(): Supplied non-integer mask'
                              ' with dtype={0}'.format(mask.dtype))

            self.mask = mask.astype('uint8')

    def set_node_id(self, node_id):
        # type: (int) -> None
        """Changes the node_id and updates the UID with it.
        Do NOT use this unless you know what you're doing;
        changing the node_id should be (1) checked against node_id
        conflics within the doc, (2) reflected in the outlinks
        and inlinks.
        """
        self.node_id = node_id
        self._sync_node_id_to_unique_id()

    def _sync_node_id_to_unique_id(self):
        # type: () -> None
        """Resets the UID number to reflect the node_id."""
        global_name, document_name, node_id = self._parse_uid(self.unique_id)
        unique_id = self.build_unique_id(global_name, document_name, self.node_id)
        self.set_unique_id(unique_id)

    @property
    def dataset(self):
        """Which dataset is this Node coming from?
        For bookkeeping."""
        # The ``_dataset_namespace`` is set during initialization.
        return self._dataset_namespace

    @property
    def doc(self):
        """Which document within the dataset is this Node
        coming from? The ``_document_namespace``

        This is important when working with CropObjects
        from multiple CropObjectList files, especially for properly
        constructing Node graphs, because ``inlinks`` and
        ``outlinks`` use the numeric ``objids``, which point to
        CropObjects within the same document.

        ``node_id`` of each Node has to be unique within a document.
        """
        # The ``_document_namespace`` is set during initialization.
        return self._document_namespace

    @property
    def top(self):
        """Row coordinate of upper left corner."""
        return self.x

    @property
    def bottom(self):
        """Row coordinate 1 beyond bottom right corner, so that indexing
        in the form ``img[node.top:node.bottom]`` is possible."""
        return self.x + self.height

    @property
    def left(self):
        """Column coordinate of upper left corner."""
        return self.y

    @property
    def right(self):
        """Column coordinate 1 beyond bottom right corner, so that indexing
        in the form ``img[:, node.left:node.right]`` is possible."""
        return self.y + self.width

    @property
    def bounding_box(self):
        """The ``top, left, bottom, right`` tuple of the Node's
        coordinates."""
        return self.top, self.left, self.bottom, self.right

    @property
    def middle(self):
        """Returns the integer representation of where the middle
        of the Node lies, as a ``(m_vert, m_horz)`` tuple.

        The integers just get rounded down.
        """
        vmid = self.top + (self.bottom - self.top) // 2
        hmid = self.left + (self.right - self.left) // 2
        return int(vmid), int(hmid)

    @property
    def is_empty(self):
        """A Node is empty if it is composed of zero pixels.
        This is measured through the mask. CropObjects without
        a mask are assumed to be non-empty."""
        if self.mask is None:
            return False

        return self.mask.sum() == 0

    @property
    def outlink_uids(self):
        return [self.build_unique_id(self.dataset, self.doc, o) for o in self.outlinks]

    @property
    def inlink_uids(self):
        return [self.build_unique_id(self.dataset, self.doc, i) for i in self.inlinks]

    @staticmethod
    def bbox_to_integer_bounds(ftop, fleft, fbottom, fright):
        """Rounds off the Node bounds to the nearest integer
        so that no area is lost (e.g. bottom and right bounds are
        rounded up, top and left bounds are rounded down).

        Returns the rounded-off integers (top, left, bottom, right)
        as integers.

        >>> Node.bbox_to_integer_bounds(44.2, 18.9, 55.1, 92.99)
        (44, 18, 56, 93)
        >>> Node.bbox_to_integer_bounds(44, 18, 56, 92.99)
        (44, 18, 56, 93)

        """
        logging.debug('bbox_to_integer_bounds: inputs {0}'.format((ftop, fleft, fbottom, fright)))

        top = ftop - (ftop % 1.0)
        left = fleft - (fleft % 1.0)
        bottom = fbottom - (fbottom % 1.0)
        if fbottom % 1.0 != 0:
            bottom += 1.0
        right = fright - (fright % 1.0)
        if fright % 1.0 != 0:
            right += 1.0

        if top != ftop:
            logging.debug('bbox_to_integer_bounds: rounded top by {0}'.format(top - ftop))
        if left != fleft:
            logging.debug('bbox_to_integer_bounds: rounded left by {0}'.format(left - fleft))
        if bottom != fbottom:
            logging.debug('bbox_to_integer_bounds: rounded bottom by {0}'.format(bottom - fbottom))
        if right != fright:
            logging.debug('bbox_to_integer_bounds: rounded right by {0}'.format(right - fright))

        return int(top), int(left), int(bottom), int(right)

    def to_integer_bounds(self):
        """Ensures that the Node has an integer position and size.
        (This is important whenever you want to use a mask, and reasonable
        whenever you do not need sub-pixel resolution...)
        """
        bbox = self.bounding_box
        t, l, b, r = self.bbox_to_integer_bounds(*bbox)
        height = b - t
        width = r - l

        self.x = t
        self.y = l
        self.height = height
        self.width = width

    def project_to(self, img):
        """This function returns the *crop* of the input image
        corresponding to the Node (incl. masking).
        Assumes zeros are background."""
        # Make a copy! We don't want to modify the original image by the mask.
        # Copy forced by the "* 1" part.
        crop = img[self.top:self.bottom, self.left:self.right] * 1
        if self.mask is not None:
            crop *= self.mask
        return crop

    def project_on(self, img):
        """This function returns only those parts of the input image
        that correspond to the Node and masks out everything else
        with zeros. The dimension of the returned array is the same
        as of the input image. This function basically reconstructs
        the symbol as an indicator function over the pixels of
        the annotated image."""
        output = numpy.zeros(img.shape, img.dtype)
        crop = self.project_to(img)
        output[self.top:self.bottom, self.left:self.right] = crop
        return output

    def render(self, img, alpha=0.3, rgb=(1.0, 0.0, 0.0)):
        """Renders itself upon the given image as a rectangle
        of the given color and transparency. Might help visualization.

        :param img: A three-channel image (3-D numpy array,
            with the last dimension being 3)."""
        color = numpy.array(rgb)
        logging.debug('Rendering object {0}, class_name {1}, t/b/l/r: {2}'
                      ''.format(self.node_id, self.class_name,
                                (self.top, self.bottom, self.left, self.right)))
        # logging.debug('Shape: {0}'.format((self.height, self.width, 3)))
        mask = numpy.ones((self.height, self.width, 3)) * color
        crop = img[self.top:self.bottom, self.left:self.right]
        # logging.debug('Mask done, creating crop')
        logging.debug('Shape: {0}. Got crop. Crop shape: {1}, img shape: {2}'
                      ''.format((self.height, self.width, 3), crop.shape, img.shape))
        mix = (crop + alpha * mask) / (1 + alpha)

        img[self.top:self.bottom, self.left:self.right] = mix
        return img

    def overlaps(self, bounding_box_or_cropobject):
        """Check whether this Node overlaps the given bounding box or Node.

        >>> node = Node(0, 'test', 10, 100, height=20, width=10)
        >>> node.bounding_box
        (10, 100, 30, 110)
        >>> node.overlaps((10, 100, 30, 110))  # Exact match
        True
        >>> node.overlaps((0, 100, 8, 110))    # Row mismatch
        False
        >>> node.overlaps((10, 0, 30, 89))     # Column mismatch
        False
        >>> node.overlaps((0, 0, 8, 89))       # Total mismatch
        False
        >>> node.overlaps((9, 99, 31, 111))    # Encompasses Node
        True
        >>> node.overlaps((11, 101, 29, 109))
        True
        >>> node.overlaps((9, 101, 31, 109))   # Encompass horz., within vert.
        True
        >>> node.overlaps((11, 99, 29, 111))   # Encompasses vert., within horz.
        True
        >>> node.overlaps((11, 101, 31, 111))  # Corner within: top left
        True
        >>> node.overlaps((11, 99, 31, 109))   # Corner within: top right
        True
        >>> node.overlaps((9, 101, 29, 111))   # Corner within: bottom left
        True
        >>> node.overlaps((9, 99, 29, 109))    # Corner within: bottom right
        True

        """
        if isinstance(bounding_box_or_cropobject, Node):
            t, l, b, r = bounding_box_or_cropobject.bounding_box
        else:
            t, l, b, r = bounding_box_or_cropobject
        # Does it overlap vertically? Includes situations where the CropObject is inside the bounding box.
        # Note that the bottom is +1 (fencepost), so the checks bottom vs. top need to be "less than",
        # not leq. If one object's top would be equal to the other's bottom, they would be touching,
        # not overlapping.
        if max(t, self.top) < min(b, self.bottom):
            if max(l, self.left) < min(r, self.right):
                return True
        return False

    def contains(self, bounding_box_or_cropobject):
        """Check if this CropObject entiNodeins the other bounding
        box (or, the other cropobject's bounding box)."""
        if isinstance(bounding_box_or_cropobject, Node):
            t, l, b, r = bounding_box_or_cropobject.bounding_box
        else:
            t, l, b, r = bounding_box_or_cropobject

        if self.top <= t <= b <= self.bottom:
            if self.left <= l <= r <= self.right:
                return True
        return False

    def bbox_intersection(self, bounding_box):
        """Returns the sub-bounding box of this CropObject, relNodets size (so: 0,0
        is the CropObject's upNodeorner), that intersects the given bounding box.
        If the intersection is empty, returns None.

        >>> node = Node(0, 'test', 10, 100, height=20, width=10)
        >>> node.bounding_box
        (10, 100, 30, 110)
        >>> other_bbox = 20, 100, 40, 105
        >>> node.bbox_intersection(other_bbox)
        (10, 0, 20, 5)
        >>> containing_bbox = 4, 55, 44, 115
        >>> node.bbox_intersection(containing_bbox)
        (0, 0, 20, 10)
        >>> contained_bbox = 12, 102, 22, 108
        >>> node.bbox_intersection(contained_bbox)
        (2, 2, 12, 8)
        >>> non_overlapping_bbox = 0, 0, 3, 3
        >>> node.bbox_intersection(non_overlapping_bbox) is None
        True

        """
        t, l, b, r = bounding_box

        out_top = max(t, self.top)
        out_bottom = min(b, self.bottom)
        out_left = max(l, self.left)
        out_right = min(r, self.right)

        if (out_top < out_bottom) and (out_left < out_right):
            return out_top - self.top, \
                   out_left - self.left, \
                   out_bottom - self.top, \
                   out_right - self.left
        else:
            return None

    def crop_to_mask(self):
        """Crops itself to the minimum bounding box that contains all
        its pixels, as determined by its mask.

        If the mask is all zeros, does not do anything, because
        at this point, the is_empty check should be invoked anyway
        in any situation where you care whether the object is empty
        or not (e.g. delete it after trimming).

        >>> mask = numpy.zeros((20, 10))
        >>> mask[5:15, 3:8] = 1
        >>> node = Node(0, 'test', 10, 100, width=10, height=20, mask=mask)
        >>> node.bounding_box
        (10, 100, 30, 110)
        >>> node.crop_to_mask()
        >>> node.bounding_box
        (15, 103, 25, 108)
        >>> node.height, node.width
        (10, 5)

        Assumes integer bounds, which is ensured during CropObject initNode.
        """
        if self.mask is None:
            return

        if self.is_empty:
            return

        # We know the object is not empty.

        # How many rows/columns to trim from top, bottom, etc.
        trim_top = -1
        for i in range(self.mask.shape[0]):
            if self.mask[i, :].sum() != 0:
                trim_top = i
                break

        trim_left = -1
        for j in range(self.mask.shape[1]):
            if self.mask[:, j].sum() != 0:
                trim_left = j
                break

        trim_bottom = -1
        for k in range(self.mask.shape[0]):
            if self.mask[-(k + 1), :].sum() != 0:
                trim_bottom = k
                break

        trim_right = -1
        for l in range(self.mask.shape[1]):
            if self.mask[:, -(l + 1)].sum() != 0:
                trim_right = l
                break

        logging.debug('Cropobject.crop: Trimming top={0}, left={1},'
                      'bottom={2}, right={3}'
                      ''.format(trim_top, trim_left, trim_bottom, trim_right))

        # new bounding box relative to the current bounding box -- used to trim
        # the mask
        rel_t = trim_top
        rel_l = trim_left
        rel_b = self.height - trim_bottom
        rel_r = self.width - trim_right

        new_mask = self.mask[rel_t:rel_b, rel_l:rel_r] * 1

        logging.debug('Cropobject.crop: Old mask shape {0}, new mask shape {1}'
                      ''.format(self.mask.shape, new_mask.shape))

        # new bounding box, relative to image -- used to compute the CropObject's position and size
        abs_t = self.top + trim_top
        abs_l = self.left + trim_left
        abs_b = self.bottom - trim_bottom
        abs_r = self.right - trim_right

        self.x = abs_t
        self.y = abs_l
        self.height = abs_b - abs_t
        self.width = abs_r - abs_l

        self.set_mask(new_mask)

    def __str__(self):
        """Format the CropObject as string representation. See the documentation
        of :module:`mung.io` for details."""
        lines = []
        lines.append('<CropObject xml:id="{}">'.format(self.unique_id))
        lines.append('\t<Id>{0}</Id>'.format(self.node_id))
        # lines.append('\t<UniqueId>{0}</UniqueId>'.format(self.unique_id))
        lines.append('\t<ClassName>{0}</ClassName>'.format(self.class_name))
        lines.append('\t<Top>{0}</Top>'.format(self.top))
        lines.append('\t<Left>{0}</Left>'.format(self.left))
        lines.append('\t<Width>{0}</Width>'.format(self.width))
        lines.append('\t<Height>{0}</Height>'.format(self.height))

        mask_string = self.encode_mask(self.mask)
        lines.append('\t<Mask>{0}</Mask>'.format(mask_string))

        if len(self.inlinks) > 0:
            inlinks_string = ' '.join(list(map(str, self.inlinks)))
            lines.append('\t<Inlinks>{0}</Inlinks>'.format(inlinks_string))
        if len(self.outlinks) > 0:
            outlinks_string = ' '.join(list(map(str, self.outlinks)))
            lines.append('\t<Outlinks>{0}</Outlinks>'.format(outlinks_string))

        data_string = self.encode_data(self.data)
        if data_string is not None:
            lines.append('\t<Data>\n{0}\n\t</Data>'.format(data_string))

        lines.append('</CropObject>')
        return '\n'.join(lines)

    def encode_mask(self, mask, compress=False, mode='rle'):
        """Encode a binary array ``mask`` as a string, compliant
        with the CropObject formNodecation in :mod:`mung.io`.
        """
        if mode == 'rle':
            return self.encode_mask_rle(mask, compress=compress)
        elif mode == 'bitmap':
            return self.encode_mask_bitmap(mask, compress=compress)

    def encode_data(self, data):
        if self.data is None:
            return None
        if len(self.data) == 0:
            return None

        lines = []
        for k, v in list(self.data.items()):
            vtype = 'str'
            vval = v
            if isinstance(v, int):
                vtype = 'int'
                vval = str(v)
            elif isinstance(v, float):
                vtype = 'float'
                vval = str(v)
            elif isinstance(v, list):
                vtype = 'list[str]'
                if len(v) > 0:
                    if isinstance(v[0], int):
                        vtype = 'list[int]'
                    elif isinstance(v[0], float):
                        vtype = 'list[float]'
                vval = ' '.join([str(vv) for vv in v])

            line = '\t\t<DataItem key="{0}" type="{1}">{2}</DataItem>' \
                   ''.format(k, vtype, vval)
            lines.append(line)

        return '\n'.join(lines)

    def data_display_text(self):
        if self.data is None:
            return '[No data]'
        if len(self.data) == 0:
            return '[No data]'

        lines = []
        for k, v in list(self.data.items()):
            lines.append('{0}:      {1}'.format(k, v))
        return '\n'.join(lines)

    @staticmethod
    def encode_mask_bitmap(mask, compress=False):
        """Encodes the mask array in a compact form. Returns 'None' if mask
        is None. If the mask is not None, uses the following algorithm:

        * Flatten the mask (then use width and height of CropObject for
Nodereshaping).
        * Record as string, with whitespace separator
        * Compress string using gz2 (if compress=True) NOT IMPLEMENTED
        * Return resulting string
        """
        if mask is None:
            return 'None'
        # By default works in row-major order.
        # So we can just prescribe 'C' without losing data.
        mask_flat = mask.flatten(order=CROPOBJECT_MASK_ORDER)
        output = ' '.join(list(map(str, mask_flat)))
        return output

    @staticmethod
    def encode_mask_rle(mask, compress=False):
        """Encodes the mask array in Run-Length Encoding. Instead of
        having the bitmap ``0 0 1 1 1 0 0 0 1 1``, the RLE encodes
        the mask as ``0:2 1:3 0:3 1:2``. This is much more compact.

        Currently, the rows of the mask are not treated in any special
        way. The mask just gets flattened and then encoded.

        Implementation:
        """
        if mask is None:
            return 'None'
        mask_flat = mask.flatten(order=CROPOBJECT_MASK_ORDER)

        output_strings = []
        current_run_type = 0
        current_run_length = 0
        for i in mask_flat:
            if i == current_run_type:
                current_run_length += 1
            else:
                s = '{0}:{1}'.format(current_run_type, current_run_length)
                output_strings.append(s)
                current_run_type = i
                current_run_length = 1
        s = '{0}:{1}'.format(current_run_type, current_run_length)
        output_strings.append(s)
        output = ' '.join(output_strings)
        return output

    def decode_mask(self, mask_string, shape):
        """Decodes a CropObject maskNodeto a binary
        numpy array of the given shape."""
        mode = self._determine_mask_mode(mask_string)
        if mode == 'rle':
            return self.decode_mask_rle(mask_string, shape=shape)
        elif mode == 'bitmap':
            return self.decode_mask_bitmap(mask_string, shape=shape)

    def _determine_mask_mode(self, mask_string):
        """If the mask string starts with '0:' or '1:', or generally
        if it contains a non-0 or 1 symbol, assume it is RLE."""
        mode = 'bitmap'
        if len(mask_string) < 3:
            mode = 'bitmap'
        elif ':' in mask_string[:3]:
            mode = 'rle'
        return mode

    @staticmethod
    def decode_mask_bitmap(mask_string, shape):
        """Decodes the mask array from the encoded form to the 2D numpy array."""
        if mask_string == 'None':
            return None
        try:
            values = list(map(float, mask_string.split()))
        except ValueError:
            logging.info('CropObject.decoNode Cannot decode mask values:\n{0}'.format(mask_string))
            raise
        mask = numpy.array(values).reshape(shape)
        return mask

    @staticmethod
    def decode_mask_rle(mask_string, shape):
        """Decodes the mask array from the RLE-encoded form
        to the 2D numpy array.
        """
        if mask_string == 'None':
            return None

        values = []
        for kv in mask_string.split(' '):
            k_string, v_string = kv.split(':')
            k, v = int(k_string), int(v_string)
            vs = [k for _ in range(v)]
            values.extend(vs)

        mask = numpy.array(values).reshape(shape)
        return mask

    def join(self, other):
        """CropObject "addition": performs an OR on this
        and the ``other`` CropObjects' masks and bounding boxes,
        and assigns to this CropObject the result. Merges
        also the inlinks and outlinks.

        Works only if the document spaces for both CropObjects
        are the same. (Otherwise changes nothing.)

        The ``class_name`` of the ``other`` is ignored.
        """
        if self.doc != other.doc:
            logging.warning('Trying to join CropObject from'
                            ' into this CropObject from skipping.'
                            ''.format(other.doc, self.doc))
            return

        # Get combined bounding box
        nt = min(self.top, other.top)
        nl = min(self.left, other.left)
        nb = max(self.bottom, other.bottom)
        nr = max(self.right, other.right)

        nh = nb - nt
        nw = nr - nl

        # Create mask of corresponding size
        new_mask = numpy.zeros((nh, nw), dtype=self.mask.dtype)

        # Find coordinates where to paste the masks
        spt = self.top - nt  # spt = self_paste_top
        spl = self.left - nl
        opt = other.top - nt
        opl = other.left - nl

        # Paste the masks into these places
        new_mask[spt:spt + self.height, spl:spl + self.width] += self.mask
        new_mask[opt:opt + other.height, opl:opl + other.width] += other.mask

        # Normalize mask value
        new_mask[new_mask != 0] = 1

        # Assign the new variables to this CropObject
        self.x = nt
        self.y = nl
        self.height = nh
        self.width = nw
        self.mask = new_mask

        # Add inlinks and outlinks (check for multiple and self-reference)
        for o in other.outlinks:
            if (o not in self.outlinks) and (o != self.node_id):
                self.outlinks.append(o)
        for i in other.inlinks:
            if (i not in self.inlinks) and (i != self.node_id):
                self.inlinks.append(i)

    def get_outlink_objects(self, cropobjects):
        """Out of the given ``cropobject`` list, return a list
        of those to which this CropObject has Node
        Can deal with CropObjects from multiple documents.
        """
        output = []
        if len(self.outlinks) == 0:
            return output

        _outlink_set = frozenset(self.outlinks)

        for c in cropobjects:
            if c.doc != self.doc:
                continue
            if c.objid in _outlink_set:
                output.append(c)
            if len(output) == len(self.outlinks):
                break
        return output

    def get_inlink_objects(self, cropobjects):
        """Out of the given ``cropobject`` list, return a list
        of those from which this CropObject has Node        Can deal with CropObjects from multiple documents.
        """
        output = []
        if len(self.inlinks) == 0:
            return output

        _inlink_set = frozenset(self.inlinks)

        for c in cropobjects:
            if c.doc != self.doc:
                continue
            if c.objid in _inlink_set:
                output.append(c)
            if len(output) == len(self.inlinks):
                break
        return output

    def translate(self, down=0, right=0):
        """Move the cropobject down and right by the given amount of pixels."""
        self.x += down
        self.y += right

    def scale(self, zoom=1.0):
        """Re-compute the CropObject withNode scaling factor."""
        mask = self.mask * 1.0
        import skimage.transform
        new_mask_shape = max(int(self.height * zoom), 1), max(int(self.width * zoom), 1)
        new_mask = skimage.transform.resize(mask,
                                            output_shape=new_mask_shape)
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0
        new_mask = new_mask.astype('uint8')

        new_height, new_width = new_mask.shape
        new_top = int(self.top * zoom)
        new_left = int(self.left * zoom)

        self.x = new_top
        self.y = new_left
        self.height = new_height
        self.width = new_width
        self.mask = new_mask


##############################################################################
# Functions for merging CropObjects and CropObjectLists
def split_cropobject_on_connected_components(c, next_objid):
    """Split the CropObject intoNodet per connected component
    of the mask. All inlinks/outlinks are retained in all the newly
    created CropObjects, and the old object is not changed. (If there
    is only one connected component, the object is returned unchanged
    in a list of length 1.)

    An ``node_id`` must be provided at which to start numbering the newly
    created CropObjects.

    The ``data`` attribute is also retained.
    """
    mask = c.mask

    # "Safety margin"
    canvas = numpy.zeros((mask.shape[0] + 2, mask.shape[1] + 2))
    canvas[1:-1, 1:-1] = mask
    cc, labels, bboxes = compute_connected_components(canvas)

    logging.info('CropObject.spliNodecs, bboxes: {1}'.format(cc, bboxes))

    if len(bboxes) == 1:
        return [c]

    output = []

    _next_objid = next_objid
    for label, (t, l, b, r) in list(bboxes.items()):
        # Background in compute_connected_components() doesn't work?
        if label == 0:
            continue

        h = b - t
        w = r - l
        m_label = (labels == label).astype('uint8')
        m = m_label[t:b, l:r]
        top = t + c.top - 1
        left = l + c.left - 1
        objid = _next_objid
        inlinks = copy.deepcopy(c.inlinks)
        outlinks = copy.deepcopy(c.outlinks)
        data = copy.deepcopy(c.data)

        new_c = Node(objid, c.clsname, top, left, w, h,
                     inlinks=inlinks, outlinks=outlinks,
                     mask=m, data=data)
        output.append(new_c)

        _next_objid += 1

    return output


def cropobjects_merge(fr, to, clsname, objid):
    """Merge the given CropObjects with respect to the other.
    Returns the new CropObject (witNodeying any of the inputs)."""
    if fr.doc != to.doc:
        raise ValueError('Cannot merge CropObjects from different documents!'
                         ' fr: {0}, to: {1}'.format(fr.doc, to.doc))

    mt, ml, mb, mr = cropobjects_merge_bbox([fr, to])
    mh = mb - mt
    mw = mr - ml
    mmask = cropobjects_merge_mask([fr, to])
    m_inlinks, m_outlinks = cropobjects_merge_links([fr, to])

    m_doc = fr.doc
    m_dataset = fr.dataset
    m_uid = Node.build_unique_id(m_dataset, m_doc, objid)

    output = Node(objid, clsname,
                  top=mt, left=ml, height=mh, width=mw,
                  mask=mmask,
                  inlinks=m_inlinks, outlinks=m_outlinks,
                  unique_id=m_uid)
    return output


def cropobjects_merge_multiple(cropobjects, clsname, objid):
    """Merge multiple cropobjects. Does not modify any of the inputs."""
    if len(set([c.doc for c in cropobjects])) > 1:
        raise ValueError('Cannot merge CropObjects from different documents!')
    mt, ml, mb, mr = cropobjects_merge_bbox(cropobjects)
    mh, mw = mb - mt, mr - ml
    m_mask = cropobjects_merge_mask(cropobjects)
    m_inlinks, m_outlinks = cropobjects_merge_links(cropobjects)

    m_doc = cropobjects[0].doc
    m_dataset = cropobjects[0].dataset
    m_uid = Node.build_unique_id(m_dataset, m_doc, objid)

    output = Node(objid, clsname,
                  top=mt, left=ml, height=mh, width=mw,
                  mask=m_mask,
                  inlinks=m_inlinks, outlinks=m_outlinks,
                  unique_id=m_uid)
    return output


def cropobjects_merge_bbox(cropobjects):
    """Computes the bounding box of a CropObject thatNode result from merging the given list of CropObjects.
    """
    # Find extremes. This will define the output cropobject.
    t, l, b, r = numpy.inf, numpy.inf, -1, -1
    for c in cropobjects:
        t = min(t, c.top)
        l = min(l, c.left)
        b = max(b, c.bottom)
        r = max(r, c.right)

    it, il, ib, ir = int(t), int(l), int(b), int(r)
    if (it != t) or (il != l) or (ib != b) or (ir != r):
        logging.warn('Merged bounding box does not consist of integers!'
                     ' {0}'.format((t, l, b, r)))

    return it, il, ib, ir


def cropobjects_merge_mask(cropobjects, intersection=False):
    """Merges the given list of cropobjects into one. Masks are combined
    by an OR operation.

    >>> c1 = Node(0, 'name', 10, 10, 4, 1, mask=numpy.ones((1, 4), dtype='uint8'))
    >>> c2 = Node(1, 'name', 11, 10, 6, 1, mask=numpy.ones((1, 6), dtype='uint8'))
    >>> c3 = Node(2, 'name', 9, 14,  2, 4, mask=numpy.ones((4, 2), dtype='uint8'))
    >>> nodes = [c1, c2, c3]
    >>> m1 = cropobjects_merge_mask(nodes)
    >>> m1.shape
    (4, 6)
    >>> print(m1)
    [[0 0 0 0 1 1]
     [1 1 1 1 1 1]
     [1 1 1 1 1 1]
     [0 0 0 0 1 1]]

    Mask behavior: if at least one of the cropobjects has a mask, then
    masking behavior is activated. The masks are combined using OR: any
    pixel of the resulting merged cropobject that corresponds to a True
    mask pixel in one of the input cropobjects will get a True mask value,
    all others (ie. including all intermediate areas) will get a False.

    If no input cropobject has a mask, then the resulting cropobject
    also will not have a mask.

    If some cropobjects have masks and some don't, fails.

    :param intersection: Instead of a union, return the mask
        intersection: only those pixels which are common to all
        the cropobjects.
    """
    # No mask
    if len([c for c in cropobjects if c.mask is not None]) == 0:
        return None

    # Some masked, some not
    for c in cropobjects:
        if c.mask is None:
            raise ValueError('Cannot deal with a mix of masked and non-masked cropobjects.')

    # Now we know all have masks.
    t, l, b, r = cropobjects_merge_bbox(cropobjects)
    h = b - t
    w = r - l
    output_mask = numpy.zeros((h, w), dtype=cropobjects[0].mask.dtype)
    for c in cropobjects:
        ct, cl, cb, cr = c.top - t, c.left - l, h - (b - c.bottom), w - (r - c.right)
        output_mask[ct:cb, cl:cr] += c.mask

    if intersection:
        output_mask[output_mask < len(cropobjects)] = 0
        output_mask[output_mask != 0] = 1
    else:
        output_mask[output_mask > 0] = 1
    return output_mask


def cropobjects_merge_links(cropobjects):
    """Collect all inlinks and outlinks of the given set of CropObjects
    to CropObjects outside of this set. The rationale for this is that
    these given ``cropobjects`` will be merged into one, so relationships
    within the set would become loops and disappear.

    (Note that this is not sufficient to update the relationships upon
    a merge, because the affected CropObjects *outside* the given set
    will need to have their inlinks/outlinks redirected to the new object.)

    :returns: A tuple of lists: ``(inlinks, outlinks)``
    """
    _internal_objids = frozenset([c.objid for c in cropobjects])
    outlinks = []
    inlinks = []
    for c in cropobjects:
        # No duplicates
        outlinks.extend([o for o in c.outlinks
                         if (o not in _internal_objids) and (o not in outlinks)])
        inlinks.extend([i for i in c.inlinks
                        if (i not in _internal_objids) and (i not in inlinks)])
    return inlinks, outlinks


def merge_cropobject_lists(*cropobject_lists):
    """Combines the CropObject listNodeferent documents
    into one list, so that inlink/outlink references still work.
    This is useful only if you want to merge two documents
    into one (e.g., if your annotators worked on different "layers"
    of data, and you want to merge these annotations).

    This just means shifting the ``node_id`` (and thus inlinks
    and outlinks). It is assumed the lists pertain to the same
    image. Uses deepcopy to avoid exposing the original lists
    to modification through the merged list.

    .. warning::

        If you are ever exporting the merged list, make sure to
        set the ``unique_id`` for the outputs correctly, if you want
        to create a new document.

    .. warning::

        Currently cannot handle precedence edges.

    """
    max_objids = [max([c.objid for c in c_list]) for c_list in cropobject_lists]
    min_objids = [min([c.objid for c in c_list]) for c_list in cropobject_lists]
    shift_by = [0] + [sum(max_objids[:i]) - min_objids[i] + 1 for i in range(1, len(max_objids))]

    new_lists = []
    for clist, s in zip(cropobject_lists, shift_by):
        new_list = []
        for c in clist:
            new_c = copy.deepcopy(c)
            # UID handling
            collection, doc, _ = new_c.parse_unique_id()
            new_uid = new_c.build_unique_id(collection, doc, c.objid + s)
            new_objid = c.objid + s
            new_c.set_unique_id(new_uid)
            new_c.objid = new_objid

            # Graph handling
            new_c.inlinks = [i + s for i in c.inlinks]
            new_c.outlinks = [o + s for o in c.outlinks]

            # Should also handle precedence...?

            new_list.append(new_c)
        new_lists.append(new_list)

    output = list(itertools.chain(*new_lists))

    return output


def link_cropobjects(fr, to, check_docname=True):
    """Add a relationship from the ``fr`` CropObject
    Nodeo`` CropObject. ModNodeCropObjects
    in-place.

    If the objects are already linked, does nothing.

    :param check_docname: If set, checks for ``docname``
        match and raises a ValueError if the CropObjects
        come from different documents.
    """
    if fr.doc != to.doc:
        if check_docname:
            raise ValueError('Cannot link two CropObjects that are')
        else:
            logging.warning('Attempting to link CropObjects from two different'
                            ' docments. From: {0}, to: {1}'
                            ''.format(fr.doc, to.doc))

    if (to.objid not in fr.outlinks) and (fr.objid in to.inlinks):
        logging.warning('Malformed object graph in document {0}:'
                        ' Relationship {1} --> {2} already exists as inlink,'
                        ' but not as outlink!.'
                        ''.format(fr.doc, fr.objid, to.objid))
    fr.outlinks.append(to.objid)
    to.inlinks.append(fr.objid)


def bbox_intersection(bbox_this, bbox_other):
    """Returns the t, l, b, r coordinates of the sub-bounding box
    of bbox_this that is also inside bbox_other.
    If the bounding boxes do not overlap, returns None."""
    t, l, b, r = bbox_other

    tt, tl, tb, tr = bbox_this

    out_top = max(t, tt)
    out_bottom = min(b, tb)
    out_left = max(l, tl)
    out_right = min(r, tr)

    if (out_top < out_bottom) and (out_left < out_right):
        return out_top - tt, \
               out_left - tl, \
               out_bottom - tt, \
               out_right - tl
    else:
        return None


def bbox_dice(bbox_this, bbox_other, vertical=False, horizontal=False):
    """Compute the Dice coefficient (intersection over union)
    for the given two bounding boxes.

    :param vertical: If set, will only return vertical IoU.

    :param horizontal: If set, will only return horizontal IoU.
        If both vertical and horizontal are set, will return
        normal IoU, as if they were both false.
    """
    t_t, t_l, t_b, t_r = bbox_this
    o_t, o_l, o_b, o_r = bbox_other

    u_t, i_t = min(t_t, o_t), max(t_t, o_t)
    u_l, i_l = min(t_l, o_l), max(t_l, o_l)
    u_b, i_b = max(t_b, o_b), min(t_b, o_b)
    u_r, i_r = max(t_r, o_r), min(t_r, o_r)

    u_vertical = max(0, u_b - u_t)
    u_horizontal = max(0, u_r - u_l)

    i_vertical = max(0, i_b - i_t)
    i_horizontal = max(0, i_r - i_l)

    if vertical and not horizontal:
        if u_vertical == 0:
            return 0
        else:
            return i_vertical / u_vertical
    elif horizontal and not vertical:
        if u_horizontal == 0:
            return 0
        else:
            return i_horizontal / u_horizontal
    else:
        if (u_horizontal == 0) or (u_vertical == 0):
            return 0
        else:
            return (i_horizontal * i_vertical) / (u_horizontal * u_vertical)


def cropobject_distance(c, d):
    """Computes the distance between two CropObjects.
    Their minimum vertical and horizontal distances are each taken
    separately, and the euclidean norm is computed from them."""
    if c.doc != d.doc:
        logging.warning('Cannot compute distances between CropObjects'
                        ' from different documents! ({0} vs. {1})'
                        ''.format(c.doc, d.doc))

    c_t, c_l, c_b, c_r = c.bounding_box
    d_t, d_l, d_b, d_r = d.bounding_box

    delta_vert = 0
    delta_horz = 0

    if (c_t <= d_t <= c_b) or (d_t <= c_t <= d_b):
        delta_vert = 0
    elif c_t < d_t:
        delta_vert = d_t - c_b
    else:
        delta_vert = c_t - d_b

    if (c_l <= d_l <= c_r) or (d_l <= c_l <= d_r):
        delta_horz = 0
    elif c_l < d_l:
        delta_horz = d_l - c_r
    else:
        delta_horz = c_l - d_r

    return numpy.sqrt(delta_vert ** 2 + delta_horz ** 2)


def cropobjects_on_canvas(cropobjects, margin=10):
    """Draws all the given CropObjects onto a zero background.
    The size of the canvas adapts to the CropObjects, with the
    given margin.

    Also returns the top left corner coordinates w.r.t. CropObjects' bboxes.
    """

    # margin is used to avoid the stafflines touching the edges,
    # which could perhaps break some assumptions down the line.
    it, il, ib, ir = cropobjects_merge_bbox(cropobjects)
    _t, _l, _b, _r = max(0, it - margin), max(0, il - margin), ib + margin, ir + margin

    canvas = numpy.zeros((_b - _t, _r - _l))

    for c in cropobjects:
        canvas[c.top - _t:c.bottom - _t, c.left - _l:c.right - _l] = c.mask * 1

    canvas[canvas != 0] = 1

    return canvas, (_t, _l)


def cropobject_mask_rpf(cropobject_gt, cropobject_pred):
    """Compute the recall, precision and f-score of the predicted
    cropobject's mask against the ground truth cropobject's mask."""
    if bbox_intersection(cropobject_gt.bounding_box,
                         cropobject_pred.bounding_box) is None:
        return 0.0, 0.0, 0.0

    mask_intersection = cropobjects_merge_mask([cropobject_gt,
                                                cropobject_pred],
                                               intersection=False)

    gt_pasted_mask = mask_intersection * 1
    t, l, b, r = cropobjects_merge_bbox([cropobject_gt, cropobject_pred])
    h, w = b - t, r - l
    ct, cl, cb, cr = cropobject_gt.top - t, \
                     cropobject_gt.left - l, \
                     h - (b - cropobject_gt.bottom), \
                     w - (r - cropobject_gt.right)
    gt_pasted_mask[ct:cb, cl:cr] += cropobject_gt.mask
    gt_pasted_mask[gt_pasted_mask != 0] = 1

    pred_pasted_mask = mask_intersection * 1
    t, l, b, r = cropobjects_merge_bbox([cropobject_pred, cropobject_pred])
    h, w = b - t, r - l
    ct, cl, cb, cr = cropobject_pred.top - t, \
                     cropobject_pred.left - l, \
                     h - (b - cropobject_pred.bottom), \
                     w - (r - cropobject_pred.right)
    pred_pasted_mask[ct:cb, cl:cr] += cropobject_pred.mask
    pred_pasted_mask[pred_pasted_mask != 0] = 1

    tp = float(mask_intersection.sum())
    fp = pred_pasted_mask.sum() - tp
    fn = gt_pasted_mask.sum() - tp
    rec, prec = tp / (tp + fn), tp / (tp + fp)
    fsc = (2 * rec * prec) / (rec + prec)

    return rec, prec, fsc
