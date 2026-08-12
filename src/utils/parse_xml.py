"""
Pascal VOC XML annotation parser.

Returns a list of dicts, one per <object>, with keys:
    label, xmin, ymin, xmax, ymax

Skips objects that are missing required fields or that have degenerate boxes
(xmin >= xmax or ymin >= ymax) and prints a warning rather than crashing the
data loading pipeline.
"""
import xml.etree.ElementTree as ET


class AnnotationParseError(ValueError):
    """Raised when an XML annotation file is unreadable or unparseable."""


def _int_or_none(el):
    if el is None or el.text is None:
        return None
    try:
        return int(float(el.text))
    except (TypeError, ValueError):
        return None


def parse_xml(xml_path):
    """Parse a Pascal VOC XML annotation file and return a list of box dicts."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise AnnotationParseError(f"Malformed XML at {xml_path}: {e}") from e
    except FileNotFoundError as e:
        raise AnnotationParseError(f"Annotation file not found: {xml_path}") from e

    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name_el = obj.find('name')
        bndbox = obj.find('bndbox')
        if name_el is None or name_el.text is None or bndbox is None:
            print(f"[parse_xml] skipping object with missing name/bndbox in {xml_path}")
            continue

        xmin = _int_or_none(bndbox.find('xmin'))
        ymin = _int_or_none(bndbox.find('ymin'))
        xmax = _int_or_none(bndbox.find('xmax'))
        ymax = _int_or_none(bndbox.find('ymax'))

        if None in (xmin, ymin, xmax, ymax):
            print(f"[parse_xml] skipping object with missing coord in {xml_path}")
            continue
        if xmin >= xmax or ymin >= ymax:
            print(
                f"[parse_xml] skipping degenerate box "
                f"({xmin}, {ymin}, {xmax}, {ymax}) in {xml_path}"
            )
            continue

        boxes.append({
            'label': name_el.text,
            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
        })

    return boxes
