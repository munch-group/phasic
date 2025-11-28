"""
Flexible state indexing system for population genetics simulations.

This module provides a dynamic, extensible system for converting between linear
indices and lineage property dictionaries. Unlike the hard-coded C++ structs,
this allows arbitrary combinations of properties to be defined at runtime.

IMPORTANT: Index 0 is reserved for the starting vertex in graph-based models and
cannot represent a combination of properties. All property combinations are indexed
starting at 1.

Example
-------
>>> # Define a two-locus state space with derived mutation tracking
>>> state_space = StateSpace([
...     Property('descendants_l1', max_value=10),
...     Property('descendants_l2', max_value=10),
...     Property('is_derived', max_value=1),
...     Property('population', max_value=2, min_value=1)
... ])
>>>
>>> # Index 0 is reserved for starting vertex
>>> state_space.index_to_props(0)
None
>>>
>>> # Convert index to properties (index 1+ map to property combinations)
>>> props = state_space.index_to_props(142)
>>> print(props)  # {'descendants_l1': 10, 'descendants_l2': 2, ...}
>>>
>>> # Convert properties to index (returns value ≥ 1)
>>> idx = state_space.props_to_index(props)
>>> print(idx)  # 142
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Property:
    """
    Defines a single property in a state space.

    A property represents one dimension of the lineage state (e.g., number of
    descendants, population label, derived allele status). Each property has
    a name, min/max values, which determines its base in the mixed-radix
    numbering system.

    Parameters
    ----------
    name : str
        Property name (e.g., 'descendants', 'population')
    max_value : int
        Maximum value this property can take
    min_value : int, optional
        Minimum value this property can take (default 0)
    offset : int, optional
        Deprecated: Use min_value instead. Minimum value offset (default 0).

    Attributes
    ----------
    base : int
        Radix base for this property: max_value - min_value + 1

    Examples
    --------
    >>> # Number of descendants (0 to sample_size)
    >>> prop = Property('descendants', max_value=10)
    >>> prop.base
    11

    >>> # Population label (1 to n_populations)
    >>> prop = Property('population', max_value=2, min_value=1)
    >>> prop.validate_value(1)  # OK
    >>> prop.validate_value(0)  # ValueError
    """

    name: str
    max_value: int
    min_value: int = 0
    offset: int = 0  # Deprecated, use min_value instead

    def __post_init__(self):
        # Handle backward compatibility: if offset is set, use it as min_value
        if self.offset != 0 and self.min_value == 0:
            object.__setattr__(self, 'min_value', self.offset)

    @property
    def base(self) -> int:
        """Radix base for mixed-radix system: max_value - min_value + 1."""
        return self.max_value - self.min_value + 1

    def validate_value(self, value: int) -> None:
        """
        Validate that a property value is in the valid range.

        Parameters
        ----------
        value : int
            Value to validate

        Raises
        ------
        ValueError
            If value is out of range [min_value, max_value]
        """
        if not (self.min_value <= value <= self.max_value):
            raise ValueError(
                f"Property '{self.name}' value {value} out of range "
                f"[{self.min_value}, {self.max_value}]"
            )

    def encode_value(self, value: int) -> int:
        """
        Encode a property value for indexing (subtract min_value).

        Parameters
        ----------
        value : int
            Property value to encode

        Returns
        -------
        int
            Encoded value (0-based)
        """
        self.validate_value(value)
        return value - self.min_value

    def decode_value(self, encoded: int) -> int:
        """
        Decode a property value from indexing (add min_value).

        Parameters
        ----------
        encoded : int
            Encoded (0-based) value

        Returns
        -------
        int
            Decoded value (with min_value)
        """
        return int(encoded + self.min_value)


class StateSpace:
    """
    Manages a collection of properties and conversions between indices and properties.

    StateSpace implements a mixed-radix numbering system where each property
    occupies a "digit" with its own base. This allows efficient conversion
    between flat integer indices and structured property dictionaries.

    IMPORTANT: Index 0 is reserved for the starting vertex in graph-based models.
    Therefore, property combinations are indexed starting at 1.

    Parameters
    ----------
    properties : List[Property]
        List of properties defining the state space, in order from least to
        most significant

    Attributes
    ----------
    properties : List[Property]
        Property definitions
    property_dict : Dict[str, Property]
        Property lookup by name
    size : int
        Total number of states in this space (including reserved index 0)

    Examples
    --------
    >>> # Single locus with population structure
    >>> space = StateSpace([
    ...     Property('descendants', max_value=10),
    ...     Property('population', max_value=2, min_value=1)
    ... ])
    >>> space.size
    34  # 11 * 2 + 1 (for starting vertex at index 0)

    >>> props = space.index_to_props(15)  # Index 15 -> actual state index 14
    >>> idx = space.props_to_index(props)
    >>> assert idx == 15
    """

    def __init__(self, properties: List[Property]) -> None:
        """
        Initialize state space with property definitions.

        Parameters
        ----------
        properties : list of Property
            List of properties defining the state space, ordered from least
            to most significant digit in the mixed-radix system.

        Raises
        ------
        ValueError
            If property names are not unique.
        """
        self.properties = properties
        self.property_dict = {p.name: p for p in properties}

        # Validate unique names
        if len(self.property_dict) != len(properties):
            raise ValueError("Property names must be unique")

        # Precompute bases for efficiency
        self._bases = np.array([p.base for p in properties])
        self._radix_powers = np.cumprod([1] + list(self._bases[:-1]))

    @property
    def size(self) -> int:
        """Total number of states in this space (including starting vertex at index 0)."""
        return int(np.prod(self._bases)) + 1  # +1 for starting vertex at index 0

    def index_to_props(
        self,
        index: Union[int, npt.NDArray[np.integer]],
        as_dict: bool = True
    ) -> Union[Dict[str, int], List[Dict[str, int]], npt.NDArray[np.integer], None]:
        """
        Convert linear index to property values.

        Note: Index 0 is reserved for the starting vertex and returns None.

        Parameters
        ----------
        index : int or ndarray of int
            Linear index or array of indices to convert.
        as_dict : bool, default=True
            If True, return dict. If False, return array of property values.

        Returns
        -------
        dict or list of dict or ndarray of int or None
            If index is 0: returns None (reserved for starting vertex).
            If scalar index and as_dict=True: dictionary mapping property names to values.
            If array index and as_dict=True: list of dictionaries (None for index 0).
            If as_dict=False: array of decoded property values (shape: (n_properties,) or (n_indices, n_properties)).

        Examples
        --------
        >>> space = StateSpace([
        ...     Property('a', max_value=2),
        ...     Property('b', max_value=2)
        ... ])
        >>> space.index_to_props(0)  # Starting vertex
        None
        >>> space.index_to_props(6)  # Index 6 -> state index 5 -> {'a': 2, 'b': 1}
        {'a': 2, 'b': 1}
        >>> space.index_to_props(6, as_dict=False)  # Returns decoded values
        array([2, 1])
        """
        # Handle array input
        if isinstance(index, np.ndarray):
            # Vectorized conversion
            indices = index - 1  # Offset by 1 (index 0 reserved for starting vertex)
            encoded = np.zeros((len(indices), len(self.properties)), dtype=int)

            for i, prop in enumerate(self.properties):
                encoded[:, i] = (indices // self._radix_powers[i]) % prop.base

            if as_dict:
                # Return list of dicts (None for index 0)
                return [
                    None if idx == 0 else {
                        p.name: p.decode_value(encoded[j, i])
                        for i, p in enumerate(self.properties)
                    }
                    for j, idx in enumerate(index)
                ]
            else:
                # Decode values for consistency with dict output
                decoded = np.zeros_like(encoded)
                for i, prop in enumerate(self.properties):
                    decoded[:, i] = prop.decode_value(encoded[:, i])
                return decoded

        # Scalar conversion
        if index == 0:
            # Index 0 is reserved for starting vertex
            return None

        remaining = index - 1  # Offset by 1 (index 0 reserved for starting vertex)
        encoded = np.zeros(len(self.properties), dtype=int)

        for i, prop in enumerate(self.properties):
            encoded[i] = (remaining // self._radix_powers[i]) % prop.base

        if as_dict:
            return {
                prop.name: prop.decode_value(encoded[i])
                for i, prop in enumerate(self.properties)
            }
        else:
            # Decode values for consistency with dict output
            return np.array([
                prop.decode_value(encoded[i])
                for i, prop in enumerate(self.properties)
            ])

    def props_to_index(
        self,
        props: Union[Dict[str, int], npt.NDArray[np.integer], None] = None,
        **kwargs: int
    ) -> Union[int, npt.NDArray[np.integer]]:
        """
        Convert property values to linear index.

        Note: Result is offset by 1 (index 0 reserved for starting vertex).

        Parameters
        ----------
        props : dict or ndarray of int or None, optional
            Property values to convert. Can be:
            - dict: mapping from property names to values
            - ndarray: property values in same order as self.properties
            - None: use kwargs instead
        **kwargs : int
            Alternative to dict: pass properties as keyword arguments.

        Returns
        -------
        int or ndarray of int
            Linear index (scalar, ≥1) or array of indices (if props is 2D array).

        Raises
        ------
        ValueError
            If both props and kwargs are specified, or neither is specified.

        Examples
        --------
        >>> space = StateSpace([
        ...     Property('a', max_value=2),
        ...     Property('b', max_value=2)
        ... ])
        >>> space.props_to_index({'a': 2, 'b': 1})
        6  # State index 5 + 1 offset
        >>> space.props_to_index(a=2, b=1)  # kwargs alternative
        6
        >>> space.props_to_index(np.array([2, 1]))  # array input (decoded values)
        6
        """
        # Handle kwargs
        if kwargs:
            if props is not None:
                raise ValueError("Cannot specify both props dict and kwargs")
            props = kwargs

        if props is None:
            raise ValueError("Must provide either props dict/array or kwargs")

        # Handle array input
        if isinstance(props, np.ndarray):
            if props.ndim == 1:
                # Single state - encode the decoded values
                encoded = np.array([
                    prop.encode_value(int(props[i]))
                    for i, prop in enumerate(self.properties)
                ])
            else:
                # Multiple states (2D array) - encode each row
                encoded = np.zeros_like(props)
                for i, prop in enumerate(self.properties):
                    encoded[:, i] = [prop.encode_value(int(val)) for val in props[:, i]]
                return np.dot(encoded, self._radix_powers) + 1  # +1 offset for starting vertex
        else:
            # Dict input
            encoded = np.array([
                prop.encode_value(props[prop.name])
                for prop in self.properties
            ])

        return int(np.dot(encoded, self._radix_powers)) + 1  # +1 offset for starting vertex


class StateVector:
    """
    Dict-like interface for lineage state with validation and conversion.

    StateVector wraps a state space and provides convenient access to properties
    via dictionary syntax or attributes. It maintains both the linear index and
    property dict representations.

    Parameters
    ----------
    state_space : StateSpace
        State space defining valid properties
    index : int, optional
        Initialize from linear index
    props : dict, optional
        Initialize from property dictionary

    Attributes
    ----------
    state_space : StateSpace
        Associated state space
    index : int
        Current linear index
    props : dict
        Current property values

    Examples
    --------
    >>> space = StateSpace([
    ...     Property('descendants', max_value=10),
    ...     Property('population', max_value=2, offset=1)
    ... ])

    >>> # Initialize from index
    >>> state = StateVector(space, index=15)
    >>> state['descendants']
    4
    >>> state['population']
    2

    >>> # Initialize from properties
    >>> state = StateVector(space, props={'descendants': 5, 'population': 1})
    >>> state.index
    5

    >>> # Modify and sync
    >>> state['descendants'] = 10
    >>> state.update_index()
    >>> state.index
    10
    """

    def __init__(
        self,
        state_space: StateSpace,
        index: Optional[int] = None,
        props: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Initialize state vector from index or properties.

        Parameters
        ----------
        state_space : StateSpace
            State space defining valid properties.
        index : int, optional
            Linear index to initialize from.
        props : dict, optional
            Property dictionary to initialize from.

        Raises
        ------
        ValueError
            If both index and props are specified, or neither is specified.
        """
        self.state_space = state_space

        if index is not None and props is not None:
            raise ValueError("Cannot specify both index and props")

        if index is not None:
            self._index = index
            self._props = state_space.index_to_props(index)
        elif props is not None:
            self._props = props.copy()
            self._index = state_space.props_to_index(props)
        else:
            raise ValueError("Must specify either index or props")

    @property
    def index(self) -> int:
        """
        Current linear index.

        Returns
        -------
        int
            Linear index in state space.
        """
        return self._index

    @property
    def props(self) -> Dict[str, int]:
        """
        Current property values.

        Returns
        -------
        dict
            Copy of property dictionary.
        """
        return self._props.copy()

    def __getitem__(self, key: str) -> int:
        """
        Get property value by name.

        Parameters
        ----------
        key : str
            Property name.

        Returns
        -------
        int
            Property value.

        Raises
        ------
        KeyError
            If property name is not found.
        """
        return self._props[key]

    def __setitem__(self, key: str, value: int) -> None:
        """
        Set property value by name.

        Note: Does not automatically update the linear index. Call update_index() after.

        Parameters
        ----------
        key : str
            Property name.
        value : int
            New property value.

        Raises
        ------
        KeyError
            If property name is not found.
        ValueError
            If value is out of valid range for this property.
        """
        if key not in self.state_space.property_dict:
            raise KeyError(f"Unknown property: {key}")
        self.state_space.property_dict[key].validate_value(value)
        self._props[key] = value

    def update_index(self) -> None:
        """
        Update linear index from current property values.

        Call this after modifying property values via __setitem__.
        """
        self._index = self.state_space.props_to_index(self._props)

    def update_props(self) -> None:
        """
        Update property values from current linear index.

        Call this after modifying the index directly.
        """
        self._props = self.state_space.index_to_props(self._index)

    def copy(self) -> 'StateVector':
        """
        Create a copy of this state vector.

        Returns
        -------
        StateVector
            New StateVector with same state space and property values.
        """
        return StateVector(self.state_space, props=self._props)

    def __repr__(self) -> str:
        """
        String representation of state vector.

        Returns
        -------
        str
            Human-readable representation showing index and properties.
        """
        props_str = ', '.join(f"{k}={v}" for k, v in self._props.items())
        return f"StateVector(index={self._index}, {props_str})"
