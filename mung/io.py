"""This module implements functions for reading and writing
the data formats used by MUSCIMA++.

Data formats
============

All MUSCIMA++ data is stored as XML, in ``<CropObject>`` elements.
These are grouped into ``<CropObjectList>`` elements, which are
the top-level elements in the ``*.xml`` dataset files.

The list of object classes used in the dataset is also stored as XML,
in ``<NodeClass>`` elements (within a ``<CropObjectClassList>``
element).

CropObject
----------

To read a CropObject list file (in this case, a test data file):

    >>> from mung.io import parse_cropobject_list
    >>> import os
    >>> file = os.path.join(os.path.dirname(__file__), '../test/test_data/01_basic.xml')
    >>> cropobjects = parse_cropobject_list(file)

The ``CropObject`` string representation is a XML object::

    <CropObject xml:id="MUSCIMA-pp_1.0___CVC-MUSCIMA_W-01_N-10_D-ideal___25">
      <Id>25</Id>
      <MLClassName>grace-notehead-full</MLClassName>
      <Top>119</Top>
      <Left>413</Left>
      <Width>16</Width>
      <Height>6</Height>
      <Selected>false</Selected>
      <Mask>1:5 0:11 (...) 1:4 0:6 1:5 0:1</Mask>
      <Outlinks>12 24 26</Outlinks>
      <Inlinks>13</Inlinks>
    </CropObject>

The CropObjects are themselves kept as a list::

    <CropObjectList>
      <CropObjects>
        <CropObject> ... </CropObject>
        <CropObject> ... </CropObject>
      </CropObjects>
    </CropObjectList>

Parsing is only implemented for files that consist of a single
``<CropObjectList>``.

Additional information
^^^^^^^^^^^^^^^^^^^^^^

.. caution::

    This part may easily be deprecated.

Arbitrary data can be added to the CropObject using the optional
``<Data>`` element. It should encode a dictionary of additional
information about the CropObject that may only apply to a subset
of CropObjects (this facultativeness is what distinguishes the
purpose of the ``<Data>`` element from just subclassing ``CropObject``).

For example, encoding the pitch, duration and precedence information
about a notehead could look like this::

    <CropObject>
        ...
        <Data>
            <DataItem key="pitch_step" type="str">D</DataItem>
            <DataItem key="pitch_modification" type="int">1</DataItem>
            <DataItem key="pitch_octave" type="int">4</DataItem>
            <DataItem key="midi_pitch_code" type="int">63</DataItem>
            <DataItem key="midi_duration" type="int">128</DataItem>
            <DataItem key="precedence_inlinks" type="list[int]">23 24 25</DataItem>
            <DataItem key="precedence_outlinks" type="list[int]">27</DataItem>
        </Data>
    </CropObject

The ``CropObject`` will then contain in its ``data`` attribute
the dictionary::

    self.data = {'pitch_step': 'D',
                 'pitch_modification': 1,
                 'pitch_octave': 4,
                 'midi_pitch_code': 63,
                 'midi_pitch_duration': 128,
                 'precedence_inlinks': [23, 24, 25],
                 'precedence_outlinks': [27]}


This is also a basic mechanism to allow you to subclass
CropObject with extra attributes without having to re-implement
parsing and export.

.. warning::

    Do not misuse this! The ``<Data>`` mechanism is primarily
    intended to encode extra information for MUSCIMarker to
    display.

Unique identification of a CropObject
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``xml:id`` is a string that uniquely identifies the CropObject
in the entire dataset. It is derived from a global dataset name and version
identifier (in this case, ``MUSCIMA++_1.0``), a CropObjectList identifier
which is unique within the dataset (derived from the filename:
usually in the format ``CVC-MUSCIMA_W-{:02}_N-{:02}_D-ideal``),
and the number of the CropObject within the given CropObjectList
(which matches the ``<Id>`` value). The delimiter is three underscores
(``___``), in order to comply with XML rules for the ``xml:id`` attribute.


Individual elements of a ``<CropObject>``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``<Id>`` is the integer ID of the CropObject inside a given
  ``<CropObjectList>`` (which generally corresponds to one XML file
  of CropObjects -- one document namespace).
* ``<MLClassName>`` is the name of the object's class (such as
  ``noteheadFull``, ``beam``, ``numeral_3``, etc.).
* ``<Top>`` is the vertical coordinate of the upper left corner of the object's
  bounding box.
* ``<Left>`` is the horizontal coordinate of the upper left corner of
  the object's bounding box.
* ``<Width>``: the amount of rows that the CropObject spans.
* ``<Height>``: the amount of columns that the CropObject spans.
* ``<Mask>``: a run-length-encoded binary (0/1) array that denotes the area
  within the CropObject's bounding box (specified by ``top``, ``left``,
  ``height`` and ``width``) that the CropObject actually occupies. If
  the mask is not given, the object is understood to occupy the entire
  bounding box. For the representation, see Implementation notes
  below.
* ``<Inlinks>``: whitespace-separate ``node_id`` list, representing CropObjects
  **from** which a relationship leads to this CropObject. (Relationships are
  directed edges, forming a directed graph of CropObjects.) The objids are
  valid in the same scope as the CropObject's ``node_id``: don't mix
  CropObjects from multiple scopes (e.g., multiple CropObjectLists)!
  If you are using CropObjects from multiple CropObjectLists at the same
  time, make sure to check against the ``unique_id``s.
* ``<Outlinks>``: whitespace-separate ``node_id`` list, representing CropObjects
  **to** which a relationship leads to this CropObject. (Relationships are
  directed edges, forming a directed graph of CropObjects.) The objids are
  valid in the same scope as the CropObject's ``node_id``: don't mix
  CropObjects from multiple scopes (e.g., multiple CropObjectLists)!
  If you are using CropObjects from multiple CropObjectLists at the same
  time, make sure to check against the ``unique_id``s.
* ``<Data>``: a list of ``<DataItem>`` elements. The elements have
  two attributes: ``key``, and ``type``. The ``key`` is what the item
  should be called in the ``data`` dict of the loaded CropObject.
  The ``type`` attribute encodes the Python type of the item and gets
  applied to the text of the ``<DataItem>`` to produce the value.
  Currently supported types are ``int``, ``float``, and ``str``,
  and ``list[int]``, ``list[float]`` and ``list[str]``. The lists
  are whitespace-separated.

The parser function provided for CropObjects does *not* check against
the presence of other elements. You can extend CropObjects for your
own purposes -- but you will have to implement parsing.

Legacy issues with X, Y, and positions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Formerly, instead of ``<Top>`` and ``<Left>``, there was a different way
of marking CropObject position:

* ``<X>`` was the HORIZONTAL coordinate of the object's upper left corner.
* ``<Y>`` was the VERTICAL coordinate of the object's upper left corner.

Due to legacy issues, the ``<X>`` in the XML file recorded the horizontal
position (column) and ``<Y>`` recorded the vertical position (row). However,
a ``CropObject`` instance uses these attributes in the more natural sense:
``cropobject.x`` is the **top** coordinate, ``cropobject.y`` is the **bottom**
coordinate.

This was unfortunate, and mostly caused by ambiguity of what X and Y mean.
So, the definition of the XML changed: instead of storing nondescript letters,
we will use tags ``<Top>`` and ``<Left>``. Note that we also swapped the order:
where previously the ordering was ``<X>`` (left) first and ``<Y>`` (top)
second, we make ``<Top>`` first and ``<Left>`` second. This corresponds
to how 2-D numpy arrays are indexed: row first, column second.

You may still run into CropObjectList files that use ``<X>`` and ``<Y>``.
The function for reading CropObjectList files, ``parse_cropobject_list()``,
can deal with it and correctly assign the coordinates, but the CropObjects
will be exported with ``<Top>`` and ``<Left>``. (This may break some
there-and-back reencoding tests.)

Implementation notes on the mask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mask is a numpy array that will be saved using run-length encoding.
The numpy array is first flattened, then runs of successive 0's and 1's
are encoded as e.g. ``0:10 `` for a run of 10 zeros.

How much space does this take?

Objects tend to be relatively convex, so after flattening, we can expect
more or less two runs per row (flattening is done in ``C`` order). Because
each run takes (approximately) 5 characters, each mask takes roughly ``5 * n_rows``
bytes to encode. This makes it efficient for objects wider than 5 pixels, with
a compression ratio approximately ``n_cols / 5``.
(Also, the numpy array needs to be made C-contiguous for that, which
explains the `CROPOBJECT_MASK_ORDER='C'` hack in `set_mask()`.)


NodeClass
---------------

This is what a single NodeClass element might look like::

    <NodeClass>
        <Id>1</Id>
        <Name>notehead-empty</Name>
        <GroupName>note-primitive/notehead-empty</GroupName>
        <Color>#FF7566</Color>
        </NodeClass>

See e.g. ``test/test_data/mff-muscima-classes-annot.xml``,
which is incidentally the real NodeClass list used
for annotating MUSCIMA++.

Similarly to a ``<CropObjectList>``, the ``<NodeClass>``
elements are organized inside a ``<CropObjectClassList>``::

   <CropObjectClassList>
      <CropObjectClasses>
        <NodeClass> ... </NodeClass>
        <NodeClass> ... </NodeClass>
      </CropObjectClasses>
    </CropObjectClassesList>

The :class:`NodeClass` represents one possible :class:`CropObject`
symbol class, such as a notehead or a time signature. Aside from defining
the "vocabulary" of available object classes for annotation, it also contains
some information about how objects of the given class should
be displayed in the MUSCIMarker annotation software (ordering
related object classes together in menus, implementing a sensible
color scheme, etc.). There is nothing interesting about this class,
we pulled it into the ``mung`` package because the object
grammar (i.e. which relationships are allowed and which are not)
depends on having NodeClass object as its "vocabulary",
and you will probably want to manipulate the data somehow based
on the objects' relationships (like reassembling notes from notation
primitives: notehead plus stem plus flags...), and the grammar
file is a reference for doing that.

"""
from __future__ import print_function, unicode_literals, division

from builtins import str
from builtins import map
import copy
import logging
import os

import collections
from typing import List

from lxml import etree

from mung.node import Node
from mung.node_class import NodeClass

__version__ = "1.0"
__author__ = "Jan Hajic jr."


##############################################################################


def parse_cropobject_list(filename):
    """From a xml file with a CropObjectList as the top element, parse
    a list of CropObjects. (See ``CropObject`` class documentation
    for a description of the XMl format.)

    Let's test whether the parsing function works:

    >>> test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    ...                              'test', 'test_data',
    ...                              'cropobjects_xy_vs_topleft')
    >>> clfile = os.path.join(test_data_dir, '01_basic_topleft.xml')
    >>> cropobjects = parse_cropobject_list(clfile)
    >>> len(cropobjects)
    48

    This parsing function can deal with the old-style CropObject XML
    that uses ``<Y>`` and ``<X>`` to index the top left corner:

    >>> clfile_xy = os.path.join(test_data_dir, '01_basic_xy.xml')
    >>> cropobjects_xy = parse_cropobject_list(clfile_xy)
    >>> len(cropobjects_xy)
    48
    >>> len(cropobjects) == len(cropobjects_xy)
    True

    However, the CropObjects export themselves back only with ``<Top>``
    and ``<Left>``.

    >>> export_xy = export_cropobject_list(cropobjects_xy)
    >>> with open(clfile) as hdl:
    ...     raw_data_topleft = '\\n'.join([l.rstrip() for l in hdl])
    >>> raw_data_topleft == export_xy
    True

    Note that what is Y in the data gets translated to cropobj.x (vertical),
    what is X gets translated to cropobj.y (horizontal).

    Let's also test the ``data`` attribute:

    >>> clfile_data = os.path.join(test_data_dir, '..', '01_basic_binary.xml')
    >>> cropobjects = parse_cropobject_list(clfile_data)
    >>> cropobjects[0].data['pitch_step']
    'G'
    >>> cropobjects[0].data['midi_pitch_code']
    79
    >>> cropobjects[0].data['precedence_outlinks']
    [8, 17]

    :returns: A list of ``CropObject``s.
    """
    tree = etree.parse(filename)
    root = tree.getroot()
    logging.debug('XML parsed.')
    cropobject_list = []

    for i, cropobject in enumerate(root.iter('CropObject')):
        ######################################################
        # Parsing one CropObject
        logging.debug('Parsing CropObject {0}'.format(i))

        # The problem with changing node_id: it is often used as
        # an integer in MUSCIMarker...
        objid = int(float(cropobject.findall('Id')[0].text))

        # ...So: we introduce UniqueId (unique_id). But there needs
        # to be some transition, to deal with files that don't
        # have it.
        try:
            try:
                # This is ugly, but the xml: namespace is not in the
                # cropobject.nsmap dict...
                uid = cropobject.attrib['{http://www.w3.org/XML/1998/namespace}node_id']
            except KeyError:
                # Fallback
                uid = cropobject.findall('UniqueId')[0].text
        except:
            uid = None
            # Generate a default UID at this level?
            # No: leave it to the CropObject class default UID mechanism.

        # Dealing with class_name transition: there shoud be no more
        # CropObjectLists without a class_name. The node_id has been
        # removed from CropObject specification, anyway.
        # class_name=None
        # _has_clsname = False
        if len(cropobject.findall('MLClassName')) > 0:
            clsname = cropobject.findall('MLClassName')[0].text
        elif len(cropobject.findall('ClassName')) > 0:
            clsname = cropobject.findall('ClassName')[0].text
        else:
            raise ValueError('CropObject {0}: no class_name provided.'.format(objid))

        #################################
        # Top left corner position

        # Helper functions for TopLeft/XY transition
        def _uses_xy(cropobject):
            xs = cropobject.findall('Y')
            ys = cropobject.findall('X')
            return (len(xs) > 0) and (len(ys) > 0)

        def _uses_topleft(cropobject):
            xs = cropobject.findall('Top')
            ys = cropobject.findall('Left')
            return (len(xs) > 0) and (len(ys) > 0)

        if _uses_xy(cropobject):
            ###########################################
            # DANGER! DANGER! DANGER! DANGER! DANGER! #
            #                                         #
            #      NOTE THE SWAP OF COORDINATES!      #
            #                                         #
            ###########################################
            top = float(cropobject.findall('Y')[0].text)
            left = float(cropobject.findall('X')[0].text)
        elif _uses_topleft(cropobject):
            top = float(cropobject.findall('Top')[0].text)
            left = float(cropobject.findall('Left')[0].text)
        else:
            raise KeyError('Cropobject {0} has neither Top/Left, nor Y/Y'
                           ' position information!'.format(objid))

        #################################
        # Shape
        width = float(cropobject.findall('Width')[0].text)
        height = float(cropobject.findall('Height')[0].text)

        #################################
        # Parsing the graph structure (Can deal with missing Inlinks/Outlinks)
        inlinks = []
        i_s = cropobject.findall('Inlinks')
        if len(i_s) > 0:
            i_s_text = cropobject.findall('Inlinks')[0].text
            if i_s_text is not None:  # Zero-length links
                inlinks = list(map(int, i_s_text.split(' ')))

        outlinks = []
        o_s = cropobject.findall('Outlinks')
        if len(o_s) > 0:
            o_s_text = cropobject.findall('Outlinks')[0].text
            if o_s_text is not None:
                outlinks = list(map(int, o_s_text.split(' ')))


        #################################
        # Decode the data.
        data = cropobject.findall('Data')
        data_dict = None
        if len(data) > 0:
            data = data[0]
            data_dict = {}
            for data_item in data.findall('DataItem'):
                key = data_item.get('key')
                value_type = data_item.get('type')
                value = data_item.text

                #logging.debug('Creating data entry: key={0}, type={1},'
                #              ' value={2}'.format(key, value_type, value))

                if value_type == 'int':
                    value = int(value)
                elif value_type == 'float':
                    value = float(value)
                elif value_type.startswith('list'):
                    if value is None:
                        value = []
                    else:
                        vt_factory = str
                        if value_type.endswith('[int]'):
                            vt_factory = int
                        elif value_type.endswith('[float]'):
                            vt_factory = float
                        value = list(map(vt_factory, value.split()))

                data_dict[key] = value

        #################################
        # Create the object.
        obj = Node(node_id=objid,
                   unique_id=uid,
                   class_name=clsname,
                   top=top,
                   left=left,
                   width=width,
                   height=height,
                   inlinks=inlinks,
                   outlinks=outlinks,
                   data=data_dict)

        #################################
        # Add mask.
        # We do this only after the CropObject has been created,
        # to make sure that the width & height used to reshape
        # the flattened mask reflects what is in the CropObject.
        mask = None
        m = cropobject.findall('Mask')
        if len(m) > 0:
            mask = obj.decode_mask(cropobject.findall('Mask')[0].text,
                                   shape=(obj.height, obj.width))
        obj.set_mask(mask)
        # logging.debug('Created CropObject with ID {0}'.format(obj.node_id))

        cropobject_list.append(obj)

    logging.debug('CropObjectList loaded.')

    if not validate_cropobjects_graph_structure(cropobject_list):
        raise ValueError('Invalid CropObject graph structure! Check warnings'
                         ' in log for the individual errors.')

    return cropobject_list


def validate_cropobjects_graph_structure(cropobjects):
    """Check that the graph defined by the ``inlinks`` and ``outlinks``
    in the given list of CropObjects is valid: no relationships
    leading from or to objects with non-existent ``node_id``s.

    Can deal with ``cropobjects`` coming from a combination
    of documents, through the CropObject ``doc`` property.
    Warns about documents which are found inconsistent.

    :param cropobjects: A list of :class:`CropObject` instances.

    :returns: ``True`` if graph is valid, ``False`` otherwise.
    """
    # Split into lists by doc
    cropobjects_by_doc = collections.defaultdict(list)
    for c in cropobjects:
        cropobjects_by_doc[c.doc].append(c)

    is_valid = True
    for doc, doc_cropobjects in list(cropobjects_by_doc.items()):
        doc_is_valid = validate_document_graph_structure(doc_cropobjects)
        if not doc_is_valid:
            logging.warning('Document {0} has invalid cropobject graph!'
                            ''.format(doc))
            is_valid = False
    return is_valid


def validate_document_graph_structure(nodes):
    # type: (List[Node]) -> bool
    """Check that the graph defined by the ``inlinks`` and ``outlinks``
    in the given list of CropObjects is valid: no relationships
    leading from or to objects with non-existent ``node_id``s.

    Checks that all the CropObjects come from one document. (Raises
    a ``ValueError`` otherwise.)

    :param nodes: A list of :class:`Node` instances.

    :returns: ``True`` if graph is valid, ``False`` otherwise.
    """
    docs = [node.doc for node in nodes]
    if len(set(docs)) != 1:
        raise ValueError('Got CropObjects from multiple documents!')

    is_valid = True
    node_ids = frozenset([node.node_id for node in nodes])
    for c in nodes:
        inlinks = c.inlinks
        for i in inlinks:
            if i not in node_ids:
                logging.warning('Invalid graph structure in CropObjectList:'
                                ' object {0} has inlink from non-existent'
                                ' object {1}'.format(c, i))
                is_valid = False

        outlinks = c.outlinks
        for o in outlinks:
            if o not in node_ids:
                logging.warning('Invalid graph structure in CropObjectList:'
                                ' object {0} has outlink to non-existent'
                                ' object {1}'.format(c, o))
                is_valid = False

    return is_valid


def export_cropobject_graph(cropobjects, validate=True):
    """Collects the inlink/outlink CropObject graph
    and returns it as a list of ``(from, to)`` edges.

    :param cropobjects: A list of CropObject instances.
        All are expected to be within one document.

    :param validate: If set, will raise a ValueError
        if the graph defined by the CropObjects is
        invalid.

    :returns: A list of ``(from, to)`` node_id pairs
        that represent edges in the CropObject graph.
    """
    if validate:
        validate_cropobjects_graph_structure(cropobjects)

    edges = []
    for c in cropobjects:
        for o in c.outlinks:
            edges.append((c.objid, o))
    return edges


def export_cropobject_list(cropobjects, docname=None, dataset_name=None):
    """Writes the CropObject data as a XML string. Does not write
    to a file -- use ``with open(output_file) as out_stream:`` etc.

    :param cropobjects: A list of CropObject instances.

    :param docname: Set the document name for all the CropObject
        unique IDs to this. If not given, no docname is applied.
        This means that either the old document identification
        stays (in case the CropObjects are loaded from a file
        with document IDs set), or the default is used (if the
        CropObjects have been newly created). If given,
        the CropObjects are first deep-copied, so that the existing
        objects' UID is not affected by the export.

    :param dataset_name: Analogous to docname.
    """
    if docname is not None:
        new_cropobjects = []
        for c in cropobjects:
            new_c = copy.deepcopy(c)
            new_c.set_doc(docname)
            new_cropobjects.append(new_c)
        cropobjects = new_cropobjects

    if dataset_name is not None:
        new_cropobjects = []
        for c in cropobjects:
            new_c = copy.deepcopy(c)
            new_c.set_dataset(dataset_name)
            new_cropobjects.append(new_c)
        cropobjects = new_cropobjects

    # This is the data string, the rest is formalities
    cropobj_string = '\n'.join([str(c) for c in cropobjects])

    lines = list()

    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<CropObjectList'
                 ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                 ' xmlns:xsd="http://www.w3.org/2001/XMLSchema">')
    lines.append('<CropObjects>')
    lines.append(cropobj_string)
    lines.append('</CropObjects>')
    lines.append('</CropObjectList>')
    return '\n'.join(lines)


##############################################################################
# Parsing NodeClass lists, mostly for grammars.

def parse_node_classes(filename):
    # type: (str) -> List[NodeClass]
    """ Extract the list of :class:`NodeClass` objects from
        an xml file with a NodeClasses as the top element and NodeClass children.
    """
    tree = etree.parse(filename)
    node_classes_xml = tree.getroot()
    node_classes = []
    for node_class_xml in node_classes_xml:
        node_class = NodeClass(class_id=int(node_class_xml.findall('Id')[0].text),
                               name=node_class_xml.findall('Name')[0].text,
                               group_name=node_class_xml.findall('GroupName')[0].text,
                               color=node_class_xml.findall('Color')[0].text)
        node_classes.append(node_class)
    return node_classes


def export_cropobject_class_list(cropobject_classes):
    """Writes the CropObject data as a XML string. Does not write
    to a file -- use ``with open(output_file) as out_stream:`` etc.

    :param cropobjects: A list of CropObject instances.
    """
    # This is the data string, the rest is formalities
    cropobject_classes_string = '\n'.join([str(c) for c in cropobject_classes])

    lines = list()

    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<CropObjectClassList'
                 ' noNamespaceSchema="mff-muscima-cropobject-classes.xsd"'
                 ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
                 ' xmlns:xsd="http://www.w3.org/2001/XMLSchema">')
    lines.append('<CropObjectClasses>')
    lines.append(cropobject_classes_string)
    lines.append('</CropObjectClasses>')
    lines.append('</CropObjectClassList>')
    return '\n'.join(lines)
