import json
import jsonschema
import pkg_resources
from pathlib import Path
import yaml
import click
import hashlib

from pyhf import tensor
from pyhf.exceptions import InvalidSpecification

SCHEMA_CACHE = {}
SCHEMA_BASE = "https://scikit-hep.org/pyhf/schemas/"
SCHEMA_VERSION = '1.0.0'

__all__ = [
    "EqDelimStringParamType",
    "citation",
    "digest",
    "load_schema",
    "options_from_eqdelimstring",
    "validate",
]


def __dir__():
    return __all__


def load_schema(schema_id, version=None):
    """
    Load a version of a schema, referenced by its identifier.

    Args:
        schema_id (:obj:`string`): The name of a schema to validate against.
        version (:obj:`string`): The version of the schema to use. If not set, the default will be the latest and greatest schema supported by this library. Default: ``None``.

    Raises:
        FileNotFoundError: if the provided ``schema_id`` cannot be found.

    Returns:
        :obj:`dict`: The loaded schema.

    Example:
        >>> import pyhf
        >>> schema = pyhf.utils.load_schema('defs.json')
        >>> type(schema)
        <class 'dict'>
        >>> schema.keys()
        dict_keys(['$schema', '$id', 'definitions'])
        >>> pyhf.utils.load_schema('defs.json', version='0.0.0')
        Traceback (most recent call last):
            ...
        FileNotFoundError: ...
    """
    global SCHEMA_CACHE
    if not version:
        version = SCHEMA_VERSION
    try:
        return SCHEMA_CACHE[f'{SCHEMA_BASE}{Path(version).joinpath(schema_id)}']
    except KeyError:
        pass

    path = pkg_resources.resource_filename(
        __name__, str(Path('schemas').joinpath(version, schema_id))
    )
    with open(path) as json_schema:
        schema = json.load(json_schema)
        SCHEMA_CACHE[schema['$id']] = schema
    return SCHEMA_CACHE[schema['$id']]


# load the defs.json as it is included by $ref
load_schema('defs.json')


def _is_array_or_tensor(checker, instance):
    """
    A helper function for allowing the validation of tensors as list types in schema validation.

    .. warning:

        This will check for valid array types using any backends that have been loaded so far.
    """
    return isinstance(instance, (list, *tensor.array_types))


def validate(spec, schema_id, *, version=None, allow_tensors=True):
    """
    Validate the provided instance, ``spec``, against the schema associated with ``schema_id``.

    Args:
        spec (:obj:`object`): An object instance to validate against a schema
        schema_id (:obj:`string`): The name of a schema to validate against. See :func:`pyhf.utils.load_schema` for more details.
        version (:obj:`string`): The version of the schema to use. See :func:`pyhf.utils.load_schema` for more details.
        allow_tensors (:obj:`bool`): A flag to enable or disable tensors as part of schema validation. If enabled, tensors in the ``spec`` will be treated like python :obj:`list`. Default: ``True``.

    Raises:
        ~pyhf.exceptions.InvalidSpecification: if the provided instance does not validate against the schema.

    Returns:
        None: if there are no errors with the provided instance

    Example:
        >>> import pyhf
        >>> model = pyhf.simplemodels.uncorrelated_background(
        ...     signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
        ... )
        >>> pyhf.utils.validate(model.spec, 'model.json')
        >>>
    """
    schema = load_schema(schema_id, version=version)
    try:
        resolver = jsonschema.RefResolver(
            base_uri=f"file://{pkg_resources.resource_filename(__name__, 'schemas/'):s}",
            referrer=schema_id,
            store=SCHEMA_CACHE,
        )

        Validator = jsonschema.Draft6Validator

        if allow_tensors:
            type_checker = Validator.TYPE_CHECKER.redefine('array', _is_array_or_tensor)
            Validator = jsonschema.validators.extend(
                Validator, type_checker=type_checker
            )

        validator = Validator(schema, resolver=resolver, format_checker=None)
        return validator.validate(spec)
    except jsonschema.ValidationError as err:
        raise InvalidSpecification(err, schema_id)


def options_from_eqdelimstring(opts):
    document = '\n'.join(
        f"{opt.split('=', 1)[0]}: {opt.split('=', 1)[1]}" for opt in opts
    )
    return yaml.safe_load(document)


class EqDelimStringParamType(click.ParamType):
    name = 'equal-delimited option'

    def convert(self, value, param, ctx):
        try:
            return options_from_eqdelimstring([value])
        except IndexError:
            self.fail(f'{value:s} is not a valid equal-delimited string', param, ctx)


def digest(obj, algorithm='sha256'):
    """
    Get the digest for the provided object. Note: object must be JSON-serializable.

    The hashing algorithms supported are in :mod:`hashlib`, part of Python's Standard Libraries.

    Example:

        >>> import pyhf
        >>> obj = {'a': 2.0, 'b': 3.0, 'c': 1.0}
        >>> pyhf.utils.digest(obj)
        'a38f6093800189b79bc22ef677baf90c75705af2cfc7ff594159eca54eaa7928'
        >>> pyhf.utils.digest(obj, algorithm='md5')
        '2c0633f242928eb55c3672fed5ba8612'
        >>> pyhf.utils.digest(obj, algorithm='sha1')
        '49a27f499e763766c9545b294880df277be6f545'

    Raises:
        ValueError: If the object is not JSON-serializable or if the algorithm is not supported.

    Args:
        obj (:obj:`jsonable`): A JSON-serializable object to compute the digest of. Usually a :class:`~pyhf.workspace.Workspace` object.
        algorithm (:obj:`str`): The hashing algorithm to use.

    Returns:
        digest (:obj:`str`): The digest for the JSON-serialized object provided and hash algorithm specified.
    """

    try:
        stringified = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf8')
    except TypeError:
        raise ValueError(
            "The supplied object is not JSON-serializable for calculating a hash."
        )
    try:
        hash_alg = getattr(hashlib, algorithm)
    except AttributeError:
        raise ValueError(
            f"{algorithm} is not an algorithm provided by Python's hashlib library."
        )
    return hash_alg(stringified).hexdigest()


def citation(oneline=False):
    """
    Get the bibtex citation for pyhf

    Example:

        >>> import pyhf
        >>> pyhf.utils.citation(oneline=True)
        '@software{pyhf,  author = {Lukas Heinrich and Matthew Feickert and Giordon Stark},  title = "{pyhf: v0.6.3}",  version = {0.6.3},  doi = {10.5281/zenodo.1169739},  url = {https://doi.org/10.5281/zenodo.1169739},  note = {https://github.com/scikit-hep/pyhf/releases/tag/v0.6.3}}@article{pyhf_joss,  doi = {10.21105/joss.02823},  url = {https://doi.org/10.21105/joss.02823},  year = {2021},  publisher = {The Open Journal},  volume = {6},  number = {58},  pages = {2823},  author = {Lukas Heinrich and Matthew Feickert and Giordon Stark and Kyle Cranmer},  title = {pyhf: pure-Python implementation of HistFactory statistical models},  journal = {Journal of Open Source Software}}'

    Keyword Args:
        oneline (:obj:`bool`): Whether to provide citation with new lines (default) or as a one-liner.

    Returns:
        citation (:obj:`str`): The citation for this software
    """
    path = Path(
        pkg_resources.resource_filename(
            __name__, str(Path('data').joinpath('citation.bib'))
        )
    )
    with path.open() as fp:
        # remove end-of-file newline if there is one
        data = fp.read().strip()

    if oneline:
        data = ''.join(data.splitlines())
    return data
