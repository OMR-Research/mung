"""This module implements functions for reading and writing the data formats used by MUSCIMA++.

Data formats
============

All MUSCIMA++ data is stored as XML, in ``<Node>`` elements.
These are grouped into ``<Nodes>`` elements, which are
the top-level elements in the ``*.xml`` dataset files.

The list of object classes used in the dataset is also stored as XML,
in ``<NodeClass>`` elements (within a ``<NodeClasses>``
element).

Node
----------

To read a Node list file (in this case, a test data file):

    >>> from mung.io import read_nodes_from_file
    >>> import os
    >>> file = os.path.join(os.path.dirname(__file__), '../test/test_data/01_basic.xml')
    >>> nodes = read_nodes_from_file(file)

The ``Node`` string representation is a XML object::

    <Node xml:id="MUSCIMA-pp_1.0___CVC-MUSCIMA_W-01_N-10_D-ideal___25">
      <Id>25</Id>
      <ClassName>grace-notehead-full</ClassName>
      <Top>119</Top>
      <Left>413</Left>
      <Width>16</Width>
      <Height>6</Height>
      <Mask>1:5 0:11 (...) 1:4 0:6 1:5 0:1</Mask>
      <Outlinks>12 24 26</Outlinks>
      <Inlinks>13</Inlinks>
    </Node>

The Nodes are themselves kept as a list::

    <Nodes>
      <Node> ... </Node>
      <Node> ... </Node>
    </Nodes>

Parsing is only implemented for files that consist of a single
``<Nodes>``.

Additional information
^^^^^^^^^^^^^^^^^^^^^^

.. caution::

    This part may easily be deprecated.

Arbitrary data can be added to the Node using the optional
``<Data>`` element. It should encode a dictionary of additional
information about the Node that may only apply to a subset
of Nodes (this facultativeness is what distinguishes the
purpose of the ``<Data>`` element from just subclassing ``Node``).

For example, encoding the pitch, duration and precedence information
about a notehead could look like this::

    <Node>
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
    </Node

The ``Node`` will then contain in its ``data`` attribute
the dictionary::

    self.data = {'pitch_step': 'D',
                 'pitch_modification': 1,
                 'pitch_octave': 4,
                 'midi_pitch_code': 63,
                 'midi_pitch_duration': 128,
                 'precedence_inlinks': [23, 24, 25],
                 'precedence_outlinks': [27]}


This is also a basic mechanism to allow you to subclass
Node with extra attributes without having to re-implement
parsing and export.

.. warning::

    Do not misuse this! The ``<Data>`` mechanism is primarily
    intended to encode extra information for MUSCIMarker to
    display.


Individual elements of a ``<Node>``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``<Id>`` is the unique integer ID of the Node inside this document
* ``<ClassName>`` is the name of the object's class (such as
  ``noteheadFull``, ``beam``, ``numeral3``, etc.).
* ``<Top>`` is the vertical coordinate of the upper left corner of the object's
  bounding box.
* ``<Left>`` is the horizontal coordinate of the upper left corner of
  the object's bounding box.
* ``<Width>``: the amount of rows that the Node spans.
* ``<Height>``: the amount of columns that the Node spans.
* ``<Mask>``: a run-length-encoded binary (0/1) array that denotes the area
  within the Node's bounding box (specified by ``top``, ``left``,
  ``height`` and ``width``) that the Node actually occupies. If
  the mask is not given, the object is understood to occupy the entire
  bounding box. For the representation, see Implementation notes
  below.
* ``<Inlinks>``: whitespace-separate ``id`` list, representing Nodes
  **from** which a relationship leads to this Node. (Relationships are
  directed edges, forming a directed graph of Nodes.) The objids are
  valid in the same scope as the Node's ``id``: don't mix
  Nodes from multiple scopes (e.g., multiple documents)!
  If you are using Nodes from multiple documents at the same
  time, make sure to check against the ``unique_id``s.
* ``<Outlinks>``: whitespace-separate ``id`` list, representing Nodes
  **to** which a relationship leads to this Node. (Relationships are
  directed edges, forming a directed graph of Nodes.) The objids are
  valid in the same scope as the Node's ``id``: don't mix
  Nodes from multiple scopes (e.g., multiple documents)!
  If you are using Nodes from multiple documents at the same
  time, make sure to check against the ``unique_id``s.
* ``<Data>``: a list of ``<DataItem>`` elements. The elements have
  two attributes: ``key``, and ``type``. The ``key`` is what the item
  should be called in the ``data`` dict of the loaded Node.
  The ``type`` attribute encodes the Python type of the item and gets
  applied to the text of the ``<DataItem>`` to produce the value.
  Currently supported types are ``int``, ``float``, and ``str``,
  and ``list[int]``, ``list[float]`` and ``list[str]``. The lists
  are whitespace-separated.

The parser function provided for Nodes does *not* check against
the presence of other elements. You can extend Nodes for your
own purposes -- but you will have to implement parsing.

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
explains the `NODE_MASK_ORDER='C'` hack in `set_mask()`.)


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

Similarly to a ``<Nodes>``, the ``<NodeClass>``
elements are organized inside a ``<NodeClasses>``::

   <NodeClasses>
      <NodeClass> ... </NodeClass>
      <NodeClass> ... </NodeClass>
   </NodeClasses>

The :class:`NodeClass` represents one possible :class:`Node`
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
import copy
import logging
import os

import collections
from typing import List

from lxml import etree

from mung.node import Node
from mung.node_class import NodeClass


def read_nodes_from_file(filename: str) -> List[Node]:
    """From a xml file with a Nodes as the top element, parse
    a list of nodes. (See ``Node`` class documentation
    for a description of the XMl format.)

    Let's test whether the parsing function works:

    >>> test_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    ...                              'test', 'test_data')
    >>> file = os.path.join(test_data_dir, '01_basic.xml')
    >>> nodes = read_nodes_from_file(file)
    >>> len(nodes)
    48

    Let's also test the ``data`` attribute:
    >>> file_with_data = os.path.join(test_data_dir, '01_basic_binary.xml')
    >>> nodes = read_nodes_from_file(file_with_data)
    >>> nodes[0].data['pitch_step']
    'G'
    >>> nodes[0].data['midi_pitch_code']
    79
    >>> nodes[0].data['precedence_outlinks']
    [8, 17]
    >>> nodes[0].dataset
    'testdataset'
    >>> nodes[0].document
    '01_basic_binary'

    :returns: A list of ``Node``s.
    """
    tree = etree.parse(filename)
    root = tree.getroot()
    logging.debug('XML parsed.')
    nodes = []
    dataset = root.attrib['dataset']
    document = root.attrib['document']

    for i, node in enumerate(root.iter('Node')):
        ######################################################
        logging.debug('Parsing Node {0}'.format(i))

        node_id = int(float(node.findall('Id')[0].text))
        class_name = node.findall('ClassName')[0].text
        top = int(node.findall('Top')[0].text)
        left = int(node.findall('Left')[0].text)
        width = int(node.findall('Width')[0].text)
        height = int(node.findall('Height')[0].text)

        #################################
        # Parsing the graph structure (Can deal with missing Inlinks/Outlinks)
        inlinks = []
        i_s = node.findall('Inlinks')
        if len(i_s) > 0:
            i_s_text = node.findall('Inlinks')[0].text
            if i_s_text is not None:  # Zero-length links
                inlinks = list(map(int, i_s_text.split(' ')))

        outlinks = []
        o_s = node.findall('Outlinks')
        if len(o_s) > 0:
            o_s_text = node.findall('Outlinks')[0].text
            if o_s_text is not None:
                outlinks = list(map(int, o_s_text.split(' ')))

        #################################
        data = node.findall('Data')
        data_dict = None
        if len(data) > 0:
            data = data[0]
            data_dict = {}
            for data_item in data.findall('DataItem'):
                key = data_item.get('key')
                value_type = data_item.get('type')
                value = data_item.text

                # logging.debug('Creating data entry: key={0}, type={1},'
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
        new_node = Node(id_=node_id,
                        class_name=class_name,
                        top=top,
                        left=left,
                        width=width,
                        height=height,
                        inlinks=inlinks,
                        outlinks=outlinks,
                        dataset=dataset,
                        document=document,
                        data=data_dict)

        #################################
        # Add mask.
        # We do this only after the Node has been created,
        # to make sure that the width & height used to reshape
        # the flattened mask reflects what is in the Node.
        mask = None
        mask_elements = node.findall('Mask')
        if len(mask_elements) > 0:
            mask = Node.decode_mask(mask_elements[0].text, shape=(new_node.height, new_node.width))
        new_node.set_mask(mask)
        nodes.append(new_node)

    logging.debug('Nodes loaded.')

    if not validate_nodes_graph_structure(nodes):
        raise ValueError('Invalid Node graph structure! Check warnings'
                         ' in log for the individual errors.')

    return nodes


def validate_nodes_graph_structure(nodes: List[Node]):
    """Check that the graph defined by the ``inlinks`` and ``outlinks``
    in the given list of CropObjects is valid: no relationships
    leading from or to objects with non-existent ``id``s.

    Can deal with ``cropobjects`` coming from a combination
    of documents, through the CropObject ``document`` property.
    Warns about documents which are found inconsistent.

    :param nodes: A list of :class:`CropObject` instances.

    :returns: ``True`` if graph is valid, ``False`` otherwise.
    """
    # Split into lists by document
    cropobjects_by_doc = collections.defaultdict(list)
    for c in nodes:
        cropobjects_by_doc[c.document].append(c)

    is_valid = True
    for doc, doc_cropobjects in list(cropobjects_by_doc.items()):
        doc_is_valid = validate_document_graph_structure(doc_cropobjects)
        if not doc_is_valid:
            logging.warning('Document {0} has invalid cropobject graph!'
                            ''.format(doc))
            is_valid = False
    return is_valid


def validate_document_graph_structure(nodes: List[Node]) -> bool:
    """Check that the graph defined by the ``inlinks`` and ``outlinks``
    in the given list of CropObjects is valid: no relationships
    leading from or to objects with non-existent ``id``s.

    Checks that all the CropObjects come from one document. (Raises
    a ``ValueError`` otherwise.)

    :param nodes: A list of :class:`Node` instances.

    :returns: ``True`` if graph is valid, ``False`` otherwise.
    """
    docs = [node.document for node in nodes]
    if len(set(docs)) != 1:
        raise ValueError('Got CropObjects from multiple documents!')

    is_valid = True
    node_ids = frozenset([node.id for node in nodes])
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

    :returns: A list of ``(from, to)`` id pairs
        that represent edges in the CropObject graph.
    """
    if validate:
        validate_nodes_graph_structure(cropobjects)

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
        if node_class_xml.tag != "NodeClass":
            continue
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
