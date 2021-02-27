import copy
import itertools
import logging
from typing import List, Union, Tuple, Optional, Any

import numpy
from math import ceil

from mung.utils import compute_connected_components


class Node(object):
    """One annotated object.

    The Node represents one instance of an annotation. It implements
    the following attributes:

    * ``node_id``: the unique number of the given annotation instance in the set
      of annotations encoded in the containing `NodeList`.
    * ``dataset``: the name of the dataset this Node belongs to, e.g., MUSCIMA++_2.0
    * ``document``: the name of the document this Node belongs to, e.g., CVC-MUSCIMA_W-05_N-19_D-ideal
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
    >>> dataset = 'MUSCIMA-pp_2.0'
    >>> document = 'CVC-MUSCIMA_W-35_N-08_D-ideal'
    >>> node = Node(611, class_name=class_name,
    ...                top=top, left=left, height=height, width=width,
    ...                inlinks=[], outlinks=[],
    ...                mask=mask,
    ...                dataset=dataset, document=document)

    Nodes can also form graphs, using the following attributes:

    * ``outlinks``: Outgoing edges. A list of integers; it is assumed they are
      valid ``node_id`` within the same global/doc namespace.
    * ``inlinks``: Incoming edges. A list of integers; it is assumed they are
      valid ``node_id`` within the same global/doc namespace.

    So far, Node graphs do not support multiple relationship types.

    **Unique identification**

    The ``unique_id`` serves to identify the Node uniquely,
    at least within the MUSCIMA dataset system. (We anticipate further
    versions of the dataset, and need to plan for that.)

    To uniquely identify a Node, there are three "levels":

    * The "global", **dataset-level identification**: which dataset is this
      Node coming from? (For this dataset: ``MUSCIMA++_1.0``)
    * The "local", **document-level identification**: which document
      (within the given dataset) is this Node coming from?
      For MUSCIMA++ 1.0, this will usually be a string like
      ``CVC-MUSCIMA_W-35_N-08_D-ideal``, derived from the filename
      under which the Nodes containing the given Node
      is stored.
    * The **within-document identification**, which is the ``node_id``.

    These three components are joined together into one string by
    a delimiter: ``___``

    The full ``unique_id`` of a Node then might look like this::
    >>> node.unique_id
    'MUSCIMA-pp_2.0___CVC-MUSCIMA_W-35_N-08_D-ideal___611'

    And it consists of these three parts:

    >>> node.document
    'CVC-MUSCIMA_W-35_N-08_D-ideal'
    >>> node.dataset
    'MUSCIMA-pp_2.0'
    >>> node.id
    611

    **Nodes and images**

    Nodes and images are not tightly bound. This is because the same
    object can apply to multiple images: in the case of the CVC-MUSCIMA dataset,
    for example, the same Nodes are present both in the full image
    and in the staff-less image. The limitation here is that Nodes
    are based on exact pixels, so in order to retain validity, the images
    must correspond to each other exactly, as "layers".

    Because Nodes do not correspond to any given image, there is
    no facility in the data format to link them to a specific one. You have to
    take care of matching Node annotations to the right images by yourself.

    The ``Node`` class implements some interactions with images.

    To recover the area corresponding to a Node `c`, use:

    >>> image = numpy.array([]) #doctest: +SKIP
    >>> if node.mask is not None: crop = image[node.top:node.bottom, node.left:node.right] * node.mask  #doctest: +SKIP
    >>> if node.mask is None: crop = image[node.top:node.bottom, node.left:node.right]               #doctest: +SKIP

    Because this is clunky, we have implemented the following to get the crop:

    >>> crop = node.project_to(image)    #doctest: +SKIP

    And to get the Node projected onto the entire image:

    >>> crop = node.project_on(image)    #doctest: +SKIP

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

    >>> import matplotlib.pyplot as plt #doctest: +SKIP
    >>> node.render(image)           #doctest: +SKIP
    >>> plt.imshow(image); plt.show() #doctest: +SKIP

    However, `Node.render()` currently does not support rendering
    the mask.

    **Disambiguating class names**

    Since the class names are present
    through the ``class_name`` attribute (``<ClassName>`` element),
    matching the list is no longer necessary for general understanding
    of the file. The NodeClasses file serves as a disambiguation tool:
    there may be multiple annotation projects that use the same names
    but maybe define them differently and use different guidelines,
    and their respective NodeClasses allow you to interpret the symbol
    names correctly, in light of the corresponding set of definitions.

    .. note::

        In MUSCIMarker, the NodeClasses is currently necessary to define
        how Nodes are displayed: their color. (All noteheads are red,
        all barlines are green, etc.) The other function, matching names
        to ``clsid``, has been superseeded by the ``class_name`` Node
        attribute.

    **Merging Nodes**

    To merge a list of Nodes into a new one, you need to:

    * Compute the new object's bounding box: ``compute_unifying_bounding_box()``
    * Compute the new object's mask: ``compute_unifying_mask()``
    * Determine the class_name and node_id of the new object.

    Since node_id and class_name of merges may depend on external settings
    and generally cannot be reliably determined from the merged
    objects themselves (e.g. the merge of a notehead and a stem
    should be a new note symbol), you need to supply them externally.
    However, the bounding box and mask can be determined. The bounding
    box is computed simply as the smallest bounding box that
    encompasses all the Nodes, and the mask is an OR operation
    over the individual masks (or None, if the Nodes don't
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

    # Delimits the Node UID fields (global, document namespaces, id)
    UID_DELIMITER = '___'
    DEFAULT_DATASET = 'MUSCIMA_DEFAULT_DATASET_PLACEHOLDER'
    DEFAULT_DOCUMENT = 'default-document'

    def __init__(self, id_: int,
                 class_name: str,
                 top: int,
                 left: int,
                 width: int,
                 height: int,
                 outlinks: List[int] = None,
                 inlinks: List[int] = None,
                 mask: numpy.ndarray = None,
                 dataset: str = None,
                 document: str = None,
                 data=None):
        self.__id = id_
        self.__class_name = class_name
        self.__top = top
        self.__left = left
        self.__width = width
        self.__height = height

        # The mask presupposes integer bounds.
        # Applied relative to Node bounds, not the whole image.
        self.__to_integer_bounds()
        self.__mask = None
        self.set_mask(mask)

        if inlinks is None:
            inlinks = []
        self.inlinks = inlinks  # type: List[int]

        if outlinks is None:
            outlinks = []
        self.outlinks = outlinks  # type: List[int]

        if dataset is None:
            dataset = self.DEFAULT_DATASET
        self.__dataset = dataset

        if document is None:
            document = self.DEFAULT_DOCUMENT
        self.__document = document

        self.is_selected = False

        if data is None:
            data = dict()
        self.data = data

    @property
    def unique_id(self) -> str:
        """Returns the ``unique_id`` of this Node

        >>> node = Node(0, "", 0, 0, 0, 0)
        >>> node.unique_id
        'MUSCIMA_DEFAULT_DATASET_PLACEHOLDER___default-document___0'
        """
        return self.UID_DELIMITER.join([self.dataset,
                                        self.document,
                                        str(self.id)])

    @staticmethod
    def parse_unique_id(uid: str) -> (str, str, int):
        """Parse a unique identifier. This breaks down the UID into the dataset name,
        document name, and id

        The delimiter is expected to be ``___``
        (kept as ``Node.UID_DELIMITER``)

        >>> Node.parse_unique_id('MUSCIMA++_2.0___CVC-MUSCIMA_W-05_N-19_D-ideal___424')
        ('MUSCIMA++_2.0', 'CVC-MUSCIMA_W-05_N-19_D-ideal', 424)

        :returns: ``global_namespace, document_namespace, id`` triplet.
            The namespaces are strings, ``id`` is an integer. If ``unique_id``
            is ``None``, returns ``None`` as ``id`` and expects it
            to be filled in from the caller Node instance.
        """
        if uid is None:
            global_namespace = Node.DEFAULT_DATASET
            document_namespace = Node.DEFAULT_DOCUMENT
            node_id = None
        else:
            global_namespace, document_namespace, node_id_string = uid.split(Node.UID_DELIMITER)
            node_id = int(node_id_string)
        return global_namespace, document_namespace, node_id

    @property
    def id(self) -> int:
        return self.__id

    def set_id(self, id_):
        self.__id = id_

    @property
    def class_name(self) -> str:
        return self.__class_name
    
    def set_class_name(self, class_name_):
        self.__class_name = class_name_

    @property
    def dataset(self) -> str:
        return self.__dataset

    @property
    def document(self) -> str:
        return self.__document

    @property
    def top(self) -> int:
        """Row coordinate of upper left corner."""
        return self.__top

    @property
    def bottom(self) -> int:
        """Row coordinate 1 beyond bottom right corner, so that indexing
        in the form ``img[node.top:node.bottom]`` is possible."""
        return self.__top + self.__height

    @property
    def left(self) -> int:
        """Column coordinate of upper left corner."""
        return self.__left

    @property
    def right(self) -> int:
        """Column coordinate 1 beyond bottom right corner, so that indexing
        in the form ``img[:, node.left:node.right]`` is possible."""
        return self.__left + self.__width

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """The ``top, left, bottom, right`` tuple of the Node's coordinates."""
        return self.top, self.left, self.bottom, self.right

    @property
    def middle(self) -> Tuple[int, int]:
        """Returns the integer representation of where the middle
        of the Node lies, as a ``(m_vert, m_horz)`` tuple.

        The integers just get rounded down.

        >>> node = Node(0,'', 10, 20, 30, 40)
        >>> node.middle
        (30, 35)
        """
        vertical_center = self.top + self.height // 2
        horizontal_center = self.left + self.width // 2
        return int(vertical_center), int(horizontal_center)

    @property
    def mask(self) -> numpy.ndarray:
        return self.__mask

    def set_mask(self, mask: numpy.ndarray):
        """Sets the Node's mask to the given array. Performs
        some compatibility checks: size, dtype (converts to ``uint8``)."""
        if mask is None:
            self.__mask = None
        else:
            # Check dimension
            t, l, b, r = self.round_bounding_box_to_integer(self.top,
                                                            self.left,
                                                            self.bottom,
                                                            self.right)
            if mask.shape != (b - t, r - l):
                raise ValueError('Mask shape {0} does not correspond'
                                 ' to integer shape {1} of Node.'
                                 ''.format(mask.shape, (b - t, r - l)))
            if str(mask.dtype) != 'uint8':
                logging.debug('Node.set_mask(): Supplied non-integer mask'
                              ' with dtype={0}'.format(mask.dtype))

            self.__mask = mask.astype('uint8')

    @staticmethod
    def round_bounding_box_to_integer(top: float, left: float, bottom: float, right: float) \
            -> (int, int, int, int):
        """Rounds off the Node bounds to the nearest integer
        so that no area is lost (e.g. bottom and right bounds are
        rounded up, top and left bounds are rounded down).

        Returns the rounded-off integers (top, left, bottom, right)
        as integers.

        >>> Node.round_bounding_box_to_integer(44.2, 18.9, 55.1, 92.99)
        (44, 18, 56, 93)
        >>> Node.round_bounding_box_to_integer(44, 18, 56, 92.99)
        (44, 18, 56, 93)

        """
        return int(top), int(left), int(ceil(bottom)), int(ceil(right))

    def project_to(self, image: numpy.ndarray):
        """This function returns the *crop* of the input image
        corresponding to the Node (incl. masking).
        Assumes zeros are background."""
        # Make a copy! We don't want to modify the original image by the mask.
        # Copy forced by the "* 1" part.
        crop = image[self.top:self.bottom, self.left:self.right] * 1
        if self.__mask is not None:
            crop *= self.__mask
        return crop

    def project_on(self, image: numpy.ndarray):
        """This function returns only those parts of the input image
        that correspond to the Node and masks out everything else
        with zeros. The dimension of the returned array is the same
        as of the input image. This function basically reconstructs
        the symbol as an indicator function over the pixels of
        the annotated image."""
        output = numpy.zeros(image.shape, image.dtype)
        crop = self.project_to(image)
        output[self.top:self.bottom, self.left:self.right] = crop
        return output

    def render(self, image: numpy.ndarray, alpha: float = 0.3,
               rgb: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> numpy.ndarray:
        """Renders itself upon the given image as a rectangle
        of the given color and transparency. Might help visualization.
        """
        color = numpy.array(rgb)
        logging.debug('Rendering object {0}, class_name {1}, t/b/l/r: {2}'
                      ''.format(self.id, self.class_name,
                                (self.top, self.bottom, self.left, self.right)))
        # logging.debug('Shape: {0}'.format((self.height, self.width, 3)))
        mask = numpy.ones((self.__height, self.__width, 3)) * color
        crop = image[self.top:self.bottom, self.left:self.right]
        # logging.debug('Mask done, creating crop')
        logging.debug('Shape: {0}. Got crop. Crop shape: {1}, img shape: {2}'
                      ''.format((self.__height, self.__width, 3), crop.shape, image.shape))
        mix = (crop + alpha * mask) / (1 + alpha)

        image[self.top:self.bottom, self.left:self.right] = mix
        return image

    def overlaps(self, bounding_box_or_node):
        # type: (Union[Tuple[int, int, int, int], Node]) -> bool
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
        >>> node.overlaps((11, 101, 29, 109))  # Within Node
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
        if isinstance(bounding_box_or_node, Node):
            t, l, b, r = bounding_box_or_node.bounding_box
        else:
            t, l, b, r = bounding_box_or_node
        # Does it overlap vertically? Includes situations where the Node is inside the bounding box.
        # Note that the bottom is +1 (fencepost), so the checks bottom vs. top need to be "less than",
        # not leq. If one object's top would be equal to the other's bottom, they would be touching,
        # not overlapping.
        if max(t, self.top) < min(b, self.bottom):
            if max(l, self.left) < min(r, self.right):
                return True
        return False

    def contains(self, bounding_box_or_node):
        # type: (Union[Tuple[int, int, int, int], Node]) -> bool
        """Check if this Node entirely contains the other bounding
        box (or, the other node's bounding box)."""
        if isinstance(bounding_box_or_node, Node):
            top, left, bottom, right = bounding_box_or_node.bounding_box
        else:
            top, left, bottom, right = bounding_box_or_node

        if self.top <= top <= bottom <= self.bottom:
            if self.left <= left <= right <= self.right:
                return True
        return False

    def bounding_box_intersection(self, bounding_box: Tuple[int, int, int, int]) \
            -> Optional[Tuple[int, int, int, int]]:
        """Returns the sub-bounding box of this Node intersecting with the given bounding box.
        If the intersection is empty, returns None.

        >>> node = Node(0, 'test', 10, 100, height=20, width=10)
        >>> node.bounding_box
        (10, 100, 30, 110)
        >>> other_bbox = 20, 100, 40, 105
        >>> node.bounding_box_intersection(other_bbox)
        (10, 0, 20, 5)
        >>> containing_bbox = 4, 55, 44, 115
        >>> node.bounding_box_intersection(containing_bbox)
        (0, 0, 20, 10)
        >>> contained_bbox = 12, 102, 22, 108
        >>> node.bounding_box_intersection(contained_bbox)
        (2, 2, 12, 8)
        >>> non_overlapping_bbox = 0, 0, 3, 3
        >>> node.bounding_box_intersection(non_overlapping_bbox) is None
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

        Assumes integer bounds, which is ensured during Node initialization.
        """
        if self.__mask is None:
            return

        mask_is_empty = self.__mask.sum() == 0
        if mask_is_empty:
            return

        # We know the object is not empty.

        # How many rows/columns to trim from top, bottom, etc.
        trim_top = -1
        for i in range(self.__mask.shape[0]):
            if self.__mask[i, :].sum() != 0:
                trim_top = i
                break

        trim_left = -1
        for j in range(self.__mask.shape[1]):
            if self.__mask[:, j].sum() != 0:
                trim_left = j
                break

        trim_bottom = -1
        for k in range(self.__mask.shape[0]):
            if self.__mask[-(k + 1), :].sum() != 0:
                trim_bottom = k
                break

        trim_right = -1
        for l in range(self.__mask.shape[1]):
            if self.__mask[:, -(l + 1)].sum() != 0:
                trim_right = l
                break

        logging.debug('Node.crop: Trimming top={0}, left={1},'
                      'bottom={2}, right={3}'
                      ''.format(trim_top, trim_left, trim_bottom, trim_right))

        # new bounding box relative to the current bounding box -- used to trim
        # the mask
        rel_t = trim_top
        rel_l = trim_left
        rel_b = self.__height - trim_bottom
        rel_r = self.__width - trim_right

        new_mask = self.__mask[rel_t:rel_b, rel_l:rel_r] * 1

        logging.debug('Node.crop: Old mask shape {0}, new mask shape {1}'
                      ''.format(self.__mask.shape, new_mask.shape))

        # new bounding box, relative to image -- used to compute the Node's position and size
        abs_t = self.top + trim_top
        abs_l = self.left + trim_left
        abs_b = self.bottom - trim_bottom
        abs_r = self.right - trim_right

        self.__top = abs_t
        self.__left = abs_l
        self.__height = abs_b - abs_t
        self.__width = abs_r - abs_l

        self.set_mask(new_mask)

    def __str__(self):
        """Format the Node as string representation. See the documentation
        of :module:`mung.io` for details."""
        lines = []
        lines.append('<Node>')
        lines.append('\t<Id>{0}</Id>'.format(self.id))
        lines.append('\t<ClassName>{0}</ClassName>'.format(self.class_name)) # TODO change this if relevant for final XML notation 
        lines.append('\t<Top>{0}</Top>'.format(self.top))
        lines.append('\t<Left>{0}</Left>'.format(self.left))
        lines.append('\t<Width>{0}</Width>'.format(self.__width))
        lines.append('\t<Height>{0}</Height>'.format(self.__height))

        mask_string = self.encode_mask()
        lines.append('\t<Mask>{0}</Mask>'.format(mask_string))

        if len(self.inlinks) > 0:
            inlinks_string = ' '.join(list(map(str, self.inlinks)))
            lines.append('\t<Inlinks>{0}</Inlinks>'.format(inlinks_string))
        if len(self.outlinks) > 0:
            outlinks_string = ' '.join(list(map(str, self.outlinks)))
            lines.append('\t<Outlinks>{0}</Outlinks>'.format(outlinks_string))

        data_string = self.encode_data()
        if data_string is not None:
            lines.append('\t<Data>\n{0}\n\t</Data>'.format(data_string))

        lines.append('</Node>')
        return '\n'.join(lines)

    def encode_mask(self, mode: str = 'rle') -> str:
        """Encode a binary array ``mask`` as a string, compliant
        with the Node format specification in :mod:`mung.io`.
        """
        if mode == 'rle':
            return self.encode_mask_rle(self.mask)
        elif mode == 'bitmap':
            return self.encode_mask_bitmap(self.mask)

    def encode_data(self) -> Optional[str]:
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

    def data_display_text(self) -> str:
        if self.data is None:
            return '[No data]'
        if len(self.data) == 0:
            return '[No data]'

        lines = []
        for k, v in list(self.data.items()):
            lines.append('{0}:      {1}'.format(k, v))
        return '\n'.join(lines)

    @staticmethod
    def encode_mask_bitmap(mask: numpy.ndarray) -> str:
        """Encodes the mask array in a compact form. Returns 'None' if mask
        is None. If the mask is not None, uses the following algorithm:

        * Flatten the mask (then use width and height of Node for reshaping).
        * Record as string, with whitespace separator
        * Return resulting string
        """
        if mask is None:
            return 'None'
        # By default works in row-major order.
        # So we can just prescribe 'C' without losing data.
        mask_flat = mask.flatten(order='C')
        output = ' '.join(list(map(str, mask_flat)))
        return output

    @staticmethod
    def encode_mask_rle(mask: numpy.ndarray) -> str:
        """Encodes the mask array in Run-Length Encoding. Instead of
        having the bitmap ``0 0 1 1 1 0 0 0 1 1``, the RLE encodes
        the mask as ``0:2 1:3 0:3 1:2``. This is much more compact.

        Currently, the rows of the mask are not treated in any special
        way. The mask just gets flattened and then encoded.

        """
        if mask is None:
            return 'None'
        mask_flat = mask.flatten(order='C')

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

    @staticmethod
    def decode_mask(mask_string: str, shape) -> Optional[numpy.ndarray]:
        """Decodes a Node mask string into a binary
        numpy array of the given shape."""
        mode = Node.__determine_mask_mode(mask_string)
        if mode == 'rle':
            return Node.decode_mask_rle(mask_string, shape=shape)
        elif mode == 'bitmap':
            return Node.decode_mask_bitmap(mask_string, shape=shape)

    @staticmethod
    def __determine_mask_mode(mask_string: str):
        """If the mask string starts with '0:' or '1:', or generally
        if it contains a non-0 or 1 symbol, assume it is RLE."""
        mode = 'bitmap'
        if len(mask_string) < 3:
            mode = 'bitmap'
        elif ':' in mask_string[:3]:
            mode = 'rle'
        return mode

    @staticmethod
    def decode_mask_bitmap(mask_string: str, shape) -> Optional[numpy.ndarray]:
        """Decodes the mask array from the encoded form to the 2D numpy array."""
        if mask_string == 'None':
            return None
        try:
            values = list(map(float, mask_string.split()))
        except ValueError:
            logging.info(
                'Node.decode_mask_bitmap() Cannot decode mask values:\n{0}'.format(mask_string))
            raise
        mask = numpy.array(values).reshape(shape)
        return mask

    @staticmethod
    def decode_mask_rle(mask_string: str, shape) -> Optional[numpy.ndarray]:
        """Decodes the mask array from the RLE-encoded form
        to the 2D numpy array.
        """
        if mask_string == 'None':
            return None

        mask_flat = numpy.zeros(shape[0]*shape[1], numpy.uint8)
        index = 0
        for kv in mask_string.split(' '):
            k_string, v_string = kv.split(':')
            k, v = int(k_string), int(v_string)
            if k == 1:
                mask_flat[index:index+v] = 1
            index += v

        mask = mask_flat.reshape(shape)
        return mask

    def join(self, other):
        """Node "addition": performs an OR on this
        and the ``other`` Nodes' masks and bounding boxes,
        and assigns to this Node the result. Merges
        also the inlinks and outlinks.

        Works only if the document spaces for both Nodes
        are the same. (Otherwise changes nothing.)

        The ``class_name`` of the ``other`` is ignored.
        """
        if self.document != other.document:
            logging.warning(
                "Trying to join Node from different documents, which is forbidden. Skipping join.")
            return

        # Get combined bounding box
        new_top = min(self.top, other.top)
        new_left = min(self.left, other.left)
        new_bottom = max(self.bottom, other.bottom)
        new_right = max(self.right, other.right)

        new_height = new_bottom - new_top
        new_width = new_right - new_left

        # Create mask of corresponding size
        new_mask = numpy.zeros((new_height, new_width), dtype=self.__mask.dtype)

        # Find coordinates where to paste the masks
        spt = self.top - new_top
        spl = self.left - new_left
        opt = other.top - new_top
        opl = other.left - new_left

        # Paste the masks into these places
        new_mask[spt:spt + self.__height, spl:spl + self.__width] += self.__mask
        new_mask[opt:opt + other.height, opl:opl + other.width] += other.mask

        # Normalize mask value
        new_mask[new_mask != 0] = 1

        # Assign the new variables to this Node
        self.__top = new_top
        self.__left = new_left
        self.__height = new_height
        self.__width = new_width
        self.__mask = new_mask

        # Add inlinks and outlinks (check for multiple and self-reference)
        for o in other.outlinks:
            if (o not in self.outlinks) and (o != self.id):
                self.outlinks.append(o)
        for i in other.inlinks:
            if (i not in self.inlinks) and (i != self.id):
                self.inlinks.append(i)

    def get_outlink_objects(self, nodes):
        # type: (List[Node]) -> List[Node]
        """Out of the given ``nodes`` list, return a list
        of those to which this Node has outlinks.
        Can deal with Nodes from multiple documents.
        """
        return self.__check_nodes_that_have_links(self.outlinks, nodes)

    def get_inlink_objects(self, nodes):
        # type: (List[Node]) -> List[Node]
        """Out of the given ``nodes`` list, return a list
        of those from which this node has inlinks
        Can deal with Nodes from multiple documents.
        """
        return self.__check_nodes_that_have_links(self.inlinks, nodes)

    def __check_nodes_that_have_links(self, links, nodes):
        # type: (List[int], List[Node]) -> List[Node]
        output = []
        if len(links) == 0:
            return output

        link_set = frozenset(links)

        for node in nodes:
            if node.document != self.document:
                continue
            if node.id in link_set:
                output.append(node)
            if len(output) == len(self.inlinks):
                break
        return output

    def translate(self, down: int = 0, right: int = 0):
        """Move the Node down and right by the given amount of pixels."""
        self.__top += down
        self.__left += right

    def scale(self, zoom: float = 1.0):
        """Re-compute the Node with the given scaling factor."""
        mask = self.__mask * 1.0
        import skimage.transform
        new_mask_shape = max(int(self.__height * zoom), 1), max(int(self.__width * zoom), 1)
        new_mask = skimage.transform.resize(mask,
                                            output_shape=new_mask_shape)
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0
        new_mask = new_mask.astype('uint8')

        new_height, new_width = new_mask.shape
        new_top = int(self.top * zoom)
        new_left = int(self.left * zoom)

        self.__top = new_top
        self.__left = new_left
        self.__height = new_height
        self.__width = new_width
        self.__mask = new_mask

    def __to_integer_bounds(self):
        """Ensures that the Node has an integer position and size.
        (This is important whenever you want to use a mask, and reasonable
        whenever you do not need sub-pixel resolution...)
        """
        bounding_box = self.bounding_box
        top, left, bottom, right = self.round_bounding_box_to_integer(*bounding_box)
        height = bottom - top
        width = right - left

        self.__top = top
        self.__left = left
        self.__height = height
        self.__width = width

    def distance_to(self, node) -> Any:
        """Computes the distance between this node and another node.
        Their minimum vertical and horizontal distances are each taken
        separately, and the euclidean norm is computed from them."""
        if self.document != node.document:
            logging.warning('Cannot compute distances between Nodes'
                            ' from different documents! ({0} vs. {1})'
                            ''.format(self.document, node.document))

        if (self.top <= node.top <= self.bottom) or (node.top <= self.top <= node.bottom):
            delta_vert = 0
        elif self.top < node.top:
            delta_vert = node.top - self.bottom
        else:
            delta_vert = self.top - node.bottom

        if (self.left <= node.left <= self.right) or (node.left <= self.left <= node.right):
            delta_horz = 0
        elif self.left < node.left:
            delta_horz = node.left - self.right
        else:
            delta_horz = self.left - node.right

        return numpy.sqrt(delta_vert ** 2 + delta_horz ** 2)

    def compute_recall_precision_fscore_on_mask(self, other_node):
        # type: (Node) -> Tuple[float, float, float]
        """Compute the recall, precision and f-score of the predicted
        Node's mask against another node's mask."""

        if bounding_box_intersection(self.bounding_box, other_node.bounding_box) is None:
            return 0.0, 0.0, 0.0

        mask_intersection = compute_unifying_mask([(self), (other_node)], intersection=False)

        gt_pasted_mask = mask_intersection * 1
        t, l, b, r = compute_unifying_bounding_box([self, other_node])
        h, w = b - t, r - l
        ct, cl, cb, cr = self.top - t, \
                         self.left - l, \
                         h - (b - self.bottom), \
                         w - (r - self.right)
        gt_pasted_mask[ct:cb, cl:cr] += self.mask
        gt_pasted_mask[gt_pasted_mask != 0] = 1

        pred_pasted_mask = mask_intersection * 1
        t, l, b, r = other_node.bounding_box
        h, w = b - t, r - l
        ct, cl, cb, cr = other_node.top - t, \
                         other_node.left - l, \
                         h - (b - other_node.bottom), \
                         w - (r - other_node.right)
        pred_pasted_mask[ct:cb, cl:cr] += other_node.mask
        pred_pasted_mask[pred_pasted_mask != 0] = 1

        true_positives = float(mask_intersection.sum())
        false_positives = pred_pasted_mask.sum() - true_positives
        false_negatives = gt_pasted_mask.sum() - true_positives
        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        f_score = (2 * recall * precision) / (recall + precision)

        return recall, precision, f_score


##############################################################################


def split_node_by_its_connected_components(node: Node, next_node_id: int) -> List[Node]:
    """Split the Node into one object per connected component
    of the mask. All inlinks/outlinks are retained in all the newly
    created Nodes, and the old object is not changed.
    If there is only one connected component, the object is returned unchanged
    in a list with one entry.

    A ``id`` must be provided at which to start numbering the newly
    created Nodes.

    The ``data`` attribute is also retained.
    """
    # "Safety margin"
    canvas = numpy.zeros((node.mask.shape[0] + 2, node.mask.shape[1] + 2))
    canvas[1:-1, 1:-1] = node.mask
    number_of_connected_components, labels, bounding_boxes = compute_connected_components(canvas)

    logging.info('Node.split(): {0} connected components, bounding boxes: {1}'
                 .format(number_of_connected_components, bounding_boxes))

    if len(bounding_boxes) == 1:
        return [node]

    output = []

    for label, (top, left, bottom, right) in list(bounding_boxes.items()):
        # Background in compute_connected_components() doesn't work?
        if label == 0:
            continue

        height = bottom - top
        width = right - left
        m_label = (labels == label).astype('uint8')
        m = m_label[top:bottom, left:right]
        top = top + node.top - 1
        left = left + node.left - 1
        node_id = next_node_id
        inlinks = copy.deepcopy(node.inlinks)
        outlinks = copy.deepcopy(node.outlinks)
        data = copy.deepcopy(node.data)
        dataset = node.dataset
        document = node.document

        new_node = Node(node_id, node.class_name, top, left, width, height,
                        inlinks=inlinks, outlinks=outlinks,
                        mask=m, data=data, dataset=dataset, document=document)
        output.append(new_node)

        next_node_id += 1

    return output


def merge_nodes(first_node: Node, second_node: Node, class_name: str, id_: int) -> Node:
    """Merge the given Nodes with respect to the other.
    Returns a new Node (without modifying any of the inputs)."""
    return merge_multiple_nodes([first_node, second_node], class_name, id_)


def merge_multiple_nodes(nodes: List[Node], class_name: str, id_: int) -> Node:
    """Merge multiple nodes. Does not modify any of the inputs."""
    if len(set([c.document for c in nodes])) > 1:
        raise ValueError('Cannot merge Nodes from different documents!')
    merged_top, merged_left, merged_bottom, merged_right = compute_unifying_bounding_box(nodes)
    merged_height, merged_width = merged_bottom - merged_top, merged_right - merged_left
    merged_mask = compute_unifying_mask(nodes)
    merged_inlinks, merged_outlinks = merge_inlinks_and_outlinks_to_nodes_outside_of_this_list(
        nodes)

    dataset = nodes[0].dataset
    document = nodes[0].document

    output = Node(id_, class_name,
                  top=merged_top, left=merged_left, height=merged_height, width=merged_width,
                  mask=merged_mask,
                  inlinks=merged_inlinks, outlinks=merged_outlinks,
                  dataset=dataset, document=document)
    return output


def compute_unifying_bounding_box(nodes: List[Node]) -> (int, int, int, int):
    """ Computes the union bounding box of multiple nodes """
    top, left, bottom, right = numpy.inf, numpy.inf, -1, -1
    for node in nodes:
        top = min(top, node.top)
        left = min(left, node.left)
        bottom = max(bottom, node.bottom)
        right = max(right, node.right)

    it, il, ib, ir = int(top), int(left), int(bottom), int(right)
    if (it != top) or (il != left) or (ib != bottom) or (ir != right):
        logging.warning('Merged bounding box does not consist of integers!'
                        ' {0}'.format((top, left, bottom, right)))

    return it, il, ib, ir


def compute_unifying_mask(nodes: List[Node], intersection=False) -> Optional[numpy.ndarray]:
    """ Merges the masks of the given Nodes into one. Masks are combined by an OR operation.

    >>> c1 = Node(0, 'name', 10, 10, 4, 1, mask=numpy.ones((1, 4), dtype='uint8'))
    >>> c2 = Node(1, 'name', 11, 10, 6, 1, mask=numpy.ones((1, 6), dtype='uint8'))
    >>> c3 = Node(2, 'name', 9, 14,  2, 4, mask=numpy.ones((4, 2), dtype='uint8'))
    >>> nodes = [c1, c2, c3]
    >>> m1 = compute_unifying_mask(nodes)
    >>> m1.shape
    (4, 6)
    >>> print(m1)
    [[0 0 0 0 1 1]
     [1 1 1 1 1 1]
     [1 1 1 1 1 1]
     [0 0 0 0 1 1]]

    Mask behavior: if at least one of the Nodes has a mask, then
    masking behavior is activated. The masks are combined using OR: any
    pixel of the resulting merged Node that corresponds to a True
    mask pixel in one of the input Nodes will get a True mask value,
    all others (ie. including all intermediate areas) will get a False.

    If no input Node has a mask, then the resulting Node also will not have a mask.

    If some Nodes have masks and some don't, this call with throw an error.

    :param nodes: The list of nodes whose masks will be merged

    :param intersection: Instead of a union, return the mask
        intersection: only those pixels which are common to all
        the Nodes.
    """
    no_node_has_a_mask = len([c for c in nodes if c.mask is not None]) == 0
    if no_node_has_a_mask:
        return None

    for node in nodes:
        if node.mask is None:
            # Some nodes have masks and some don't
            raise ValueError('Cannot deal with a mix of masked and non-masked Nodes.')

    top, left, bottom, right = compute_unifying_bounding_box(nodes)
    height = bottom - top
    width = right - left
    output_mask = numpy.zeros((height, width), dtype=nodes[0].mask.dtype)
    for node in nodes:
        ct, cl, cb, cr = node.top - top, node.left - left, height - (
                bottom - node.bottom), width - (right - node.right)
        output_mask[ct:cb, cl:cr] += node.mask

    if intersection:
        output_mask[output_mask < len(nodes)] = 0
        output_mask[output_mask != 0] = 1
    else:
        output_mask[output_mask > 0] = 1
    return output_mask


def merge_inlinks_and_outlinks_to_nodes_outside_of_this_list(nodes: List[Node]) \
        -> Tuple[List[int], List[int]]:
    """Collect all inlinks and outlinks of the given set of Nodes
    to Nodes outside of this set. The rationale for this is that
    these given ``nodes`` will be merged into one, so relationships
    within the set would become loops and disappear.

    (Note that this is not sufficient to update the relationships upon
    a merge, because the affected Nodess *outside* the given set
    will need to have their inlinks/outlinks redirected to the new object.)

    :returns: A tuple of lists: ``(inlinks, outlinks)``
    """
    all_node_ids = frozenset([node.id for node in nodes])
    outlinks = []
    inlinks = []
    for c in nodes:
        # No duplicates
        outlinks.extend([o for o in c.outlinks
                         if (o not in all_node_ids) and (o not in outlinks)])
        inlinks.extend([i for i in c.inlinks
                        if (i not in all_node_ids) and (i not in inlinks)])
    return inlinks, outlinks


def merge_node_lists_from_multiple_documents(node_lists: List[List[Node]]) -> List[Node]:
    """Combines the Node lists from different documents
    into one list, so that inlink/outlink references still work.
    This is useful only if you want to merge two documents
    into one (e.g., if your annotators worked on different "layers"
    of data, and you want to merge these annotations).

    This just means shifting the ``id`` (and thus inlinks
    and outlinks). It is assumed the lists pertain to the same
    image. Uses deepcopy to avoid exposing the original lists
    to modification through the merged list.

    Currently cannot handle precedence edges.

    """
    max_node_ids = [max([node.id for node in c_list]) for c_list in node_lists]
    min_node_ids = [min([node.id for node in c_list]) for c_list in node_lists]
    shift_by = [0] + [sum(max_node_ids[:i]) - min_node_ids[i] + 1 for i in
                      range(1, len(max_node_ids))]

    new_lists = []
    for nodes, s in zip(node_lists, shift_by):
        new_list = []
        for node in nodes:
            new_node = copy.deepcopy(node)
            new_id = node.id + s
            new_node.set_id(new_id)

            # Graph handling
            new_node.inlinks = [i + s for i in node.inlinks]
            new_node.outlinks = [o + s for o in node.outlinks]

            new_list.append(new_node)
        new_lists.append(new_list)

    output = list(itertools.chain(*new_lists))

    return output


def link_nodes(from_node: Node, to_node: Node, check_that_nodes_have_the_same_document: bool = True):
    """Add a relationship from one node to the other. Updates the nodes in-place.

    If the objects are already linked, does nothing.
    """
    if from_node.document != to_node.document:
        if check_that_nodes_have_the_same_document:
            raise ValueError('Cannot link two Nodes that are')
        else:
            logging.warning('Attempting to link Nodes from two different'
                            ' docments. From: {0}, to: {1}'
                            ''.format(from_node.document, to_node.document))

    if (to_node.id not in from_node.outlinks) and (from_node.id in to_node.inlinks):
        logging.warning('Malformed object graph in document {0}:'
                        ' Relationship {1} --> {2} already exists as inlink,'
                        ' but not as outlink!.'
                        ''.format(from_node.document, from_node.id, to_node.id))
    from_node.outlinks.append(to_node.id)
    to_node.inlinks.append(from_node.id)


def bounding_box_intersection(first_bounding_box: Tuple[int, int, int, int],
                              second_bounding_box: Tuple[int, int, int, int]) -> Optional[
    Tuple[int, int, int, int]]:
    """Returns the t, l, b, r coordinates of the sub-bounding box
    of bbox_this that is also inside bbox_other.
    If the bounding boxes do not overlap, returns None."""
    t, l, b, r = second_bounding_box
    tt, tl, tb, tr = first_bounding_box

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


def bounding_box_dice_coefficient(first_bounding_box: Tuple[int, int, int, int],
                                  second_bounding_box: Tuple[int, int, int, int],
                                  vertical: bool = False,
                                  horizontal: bool = False) -> float:
    """Compute the Dice coefficient (intersection over union)
    for the given two bounding boxes.

    :param vertical: If set, will only return vertical IoU.

    :param horizontal: If set, will only return horizontal IoU.
        If both vertical and horizontal are set, will return
        normal IoU, as if they were both false.
    """
    t_t, t_l, t_b, t_r = first_bounding_box
    o_t, o_l, o_b, o_r = second_bounding_box

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
            return 0.0
        else:
            return i_vertical / u_vertical
    elif horizontal and not vertical:
        if u_horizontal == 0:
            return 0.0
        else:
            return i_horizontal / u_horizontal
    else:
        if (u_horizontal == 0) or (u_vertical == 0):
            return 0.0
        else:
            return (i_horizontal * i_vertical) / (u_horizontal * u_vertical)


def draw_nodes_on_empty_canvas(nodes: List[Node], margin: int = 10) -> Tuple[
    numpy.ndarray, Tuple[int, int]]:
    """Draws all the given Nodes onto a zero background.
    The size of the canvas adapts to the Nodes, with the    given margin.

    Also returns the top left corner coordinates w.r.t. Nodes' bounding boxes.
    """

    # margin is used to avoid the stafflines touching the edges,
    # which could perhaps break some assumptions down the line.
    top, left, bottom, right = compute_unifying_bounding_box(nodes)
    top_with_margin, left_with_margin, bottom_with_margin, right_with_margin = \
        max(0, top - margin), max(0, left - margin), bottom + margin, right + margin

    canvas = numpy.zeros(
        (bottom_with_margin - top_with_margin, right_with_margin - left_with_margin))

    for node in nodes:
        canvas[node.top - top_with_margin:node.bottom - top_with_margin,
        node.left - left_with_margin:node.right - left_with_margin] = node.mask * 1

    canvas[canvas != 0] = 1

    return canvas, (top_with_margin, left_with_margin)
