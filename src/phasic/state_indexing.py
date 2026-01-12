"""
Flexible state indexing system for population genetics simulations.

This module provides a dynamic, extensible system for converting between linear
indices and lineage property dictionaries. Unlike the hard-coded C++ structs,
this allows arbitrary combinations of properties to be defined at runtime.

Uses a mixed-radix numbering system where each property occupies a "digit" with
its own base, enabling efficient bijective mapping between flat integer indices
and structured property combinations.

Supports:
- Multiple independent PropertySets for non-overlapping state features
- Attribute-style access to PropertySets and properties
- Partial property specifications for querying/filtering states
- Backward compatibility with original StateSpace API

Example
-------
>>> # Define a property set with combinatorial properties
>>> indexer = StateIndexer(
...     lineage=[
...         Property('descendants_l1', max_value=10),
...         Property('descendants_l2', max_value=10),
...         Property('population', max_value=2, min_value=1)
...     ],
...     metadata=[
...         Property('time_bin', max_value=1000),
...         Property('branch_id', max_value=999)
...     ]
... )
>>>
>>> # Access PropertySets via attributes
>>> indexer.lineage.n_states  # 242
>>> indexer.metadata.n_states  # 1001000
>>>
>>> # Convert index to properties
>>> props = indexer.lineage.index_to_props(42)
>>> print(props)  # {'descendants_l1': 9, 'descendants_l2': 3, ...}
>>>
>>> # Backward compatible single PropertySet
>>> indexer = StateIndexer([Property('descendants', max_value=10)])
>>> indexer.n_states  # 11
>>> indexer.index_to_props(5)  # {'descendants': 5}
"""

from dataclasses import dataclass, make_dataclass
from typing import Dict, List, Optional, Union, Self
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Property:
    """
    Defines a single property in a state space.

    A property represents one dimension of the lineage state (e.g., number of
    descendants, population label, derived allele status). Each property has
    a name and min/max values for validation and encoding.

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

    Examples
    --------
    >>> # Number of descendants (0 to sample_size)
    >>> prop = Property('descendants', max_value=10)
    >>> prop.validate_value(5)  # OK
    >>> prop.validate_value(11)  # ValueError

    >>> # Population label (1 to n_populations)
    >>> prop = Property('population', max_value=2, min_value=1)
    >>> prop.validate_value(1)  # OK
    >>> prop.validate_value(0)  # ValueError
    """

    name: str
    max_value: int
    min_value: int = 0
    offset: int = 0  # Deprecated, use min_value instead

    def __post_init__(self) -> None:
        # Handle backward compatibility: if offset is set, interpret old semantics
        # Old: max_value=N, offset=M means range [M, M+N]
        # New: min_value=M, max_value=N means range [M, N]
        if self.offset != 0 and self.min_value == 0:
            # Convert old semantics to new: offset shifts the range
            object.__setattr__(self, 'min_value', self.offset)
            object.__setattr__(self, 'max_value', self.max_value + self.offset)

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
        Encode a property value to state vector element (subtract min_value).

        Parameters
        ----------
        value : int
            Property value to encode

        Returns
        -------
        int
            State vector element value (0-based offset from min_value)
        """
        self.validate_value(value)
        return value - self.min_value

    def decode_value(self, encoded: int) -> int:
        """
        Decode a state vector element to property value (add min_value).

        Parameters
        ----------
        encoded : int
            State vector element value (0-based offset)

        Returns
        -------
        int
            Property value (with min_value applied)
        """
        prop_value = encoded + self.min_value
        if hasattr(prop_value, 'size') and prop_value.size == 1:
            return int(prop_value)
        return prop_value


class PropertySet:
    """
    Collection of properties forming a combinatorial state space.

    PropertySet implements a mixed-radix numbering system where each property
    occupies a "digit" with its own base. This allows efficient conversion
    between flat integer indices and structured property dictionaries.

    Parameters
    ----------
    name : str
        Name of this property set (e.g., 'lineage', 'metadata')
    properties : List[Property]
        List of properties defining the state space, in order from least to
        most significant

    Attributes
    ----------
    name : str
        PropertySet name
    properties : List[Property]
        Property definitions
    property_dict : Dict[str, Property]
        Property lookup by name
    n_states : int
        Total number of states in this property set

    Examples
    --------
    >>> # Single locus with population structure
    >>> pset = PropertySet('lineage', [
    ...     Property('descendants', max_value=10),
    ...     Property('population', max_value=2, min_value=1)
    ... ])
    >>> pset.n_states
    22  # 11 * 2

    >>> props = pset.index_to_props(15)
    >>> idx = pset.props_to_index(props)
    >>> assert idx == 15
    """

    def __init__(self, name: str, properties: List[Property]) -> None:
        """
        Initialize property set with name and property definitions.

        Parameters
        ----------
        name : str
            Name of this property set (e.g., 'lineage', 'metadata')
        properties : list of Property
            List of properties defining the state space, ordered from least
            to most significant digit in the mixed-radix system.

        Raises
        ------
        ValueError
            If property names are not unique within this PropertySet.
        """
        self.name = name
        self.properties = properties
        self.property_dict = {p.name: p for p in properties}

        # Validate unique names
        if len(self.property_dict) != len(properties):
            raise ValueError(f"Property names must be unique within PropertySet '{name}'")

        # Precompute bases for efficiency
        self._bases = np.array([p.base for p in properties])
        self._radix_powers = np.cumprod([1] + list(self._bases[:-1]))

    @property
    def n_states(self) -> int:
        """Total number of states in this property set."""
        return int(np.prod(self._bases))

    def __repr__(self) -> str:
        """String representation of PropertySet."""
        return f"PropertySet('{self.name}', n_states={self.n_states}, properties={len(self.properties)})"

    def index_to_props(
        self,
        index: Union[int, npt.NDArray[np.integer]],
        as_dict: bool = False,
        as_values: bool = False
    ) -> Union[Dict[str, int], List[Dict[str, int]], npt.NDArray[np.integer], object, List[object]]:
        """
        Convert linear index to property values.

        Parameters
        ----------
        index : int or ndarray of int
            Linear index or array of indices to convert.
        as_dict : bool, default=False
            If True, return dict mapping property names to values.
        as_values : bool, default=False
            If True, return array of property values instead of dataclass.
            Takes precedence over as_dict.

        Returns
        -------
        dataclass or dict or list or ndarray of int
            Default (as_dict=False, as_values=False):
                - Scalar index: dataclass with properties accessible as attributes
                - Array index: list of dataclasses
            If as_dict=True:
                - Scalar index: dictionary mapping property names to values
                - Array index: list of dictionaries
            If as_values=True:
                - Scalar index: array of decoded property values
                - Array index: 2D array (n_indices, n_properties)

        Examples
        --------
        >>> pset = PropertySet('lineage', [
        ...     Property('descendants_l1', max_value=2),
        ...     Property('descendants_l2', max_value=2)
        ... ])
        >>> # Default: dataclass with attribute access
        >>> props = pset.index_to_props(5)
        >>> props.descendants_l1
        2
        >>> props.descendants_l2
        1
        >>> # As dict
        >>> pset.index_to_props(5, as_dict=True)
        {'descendants_l1': 2, 'descendants_l2': 1}
        >>> # As values array
        >>> pset.index_to_props(5, as_values=True)
        array([2, 1])
        """

        # Handle array input
        if isinstance(index, np.ndarray):
            # Vectorized conversion
            encoded = np.zeros((len(index), len(self.properties)), dtype=int)

            for i, prop in enumerate(self.properties):
                encoded[:, i] = (index // self._radix_powers[i]) % prop.base

            # Decode values
            decoded = np.zeros_like(encoded)
            for i, prop in enumerate(self.properties):
                decoded[:, i] = prop.decode_value(encoded[:, i])

            if as_values:
                return decoded
            elif as_dict:
                # Return list of dicts
                return [
                    {
                        p.name: int(decoded[j, i])
                        for i, p in enumerate(self.properties)
                    }
                    for j in range(len(index))
                ]
            else:
                # Return list of dataclasses (default)
                PropsClass = make_dataclass(
                    f'{self.name.capitalize()}Props',
                    [(p.name, int) for p in self.properties],
                    frozen=True
                )
                return [
                    PropsClass(**{
                        p.name: int(decoded[j, i])
                        for i, p in enumerate(self.properties)
                    })
                    for j in range(len(index))
                ]

        if not index < self.n_states:
            raise IndexError(f"Provided index {index} out of range for property set of size {self.n_states}")

        # Scalar conversion
        encoded = np.zeros(len(self.properties), dtype=int)

        for i, prop in enumerate(self.properties):
            encoded[i] = (index // self._radix_powers[i]) % prop.base

        # Decode values
        decoded_values = {
            prop.name: prop.decode_value(encoded[i])
            for i, prop in enumerate(self.properties)
        }

        if as_values:
            # Return array of property values
            return np.array([
                decoded_values[prop.name]
                for prop in self.properties
            ])
        elif as_dict:
            # Return dict
            return decoded_values
        else:
            # Return dataclass (default)
            PropsClass = make_dataclass(
                f'{self.name.capitalize()}Props',
                [(p.name, int) for p in self.properties],
                frozen=True
            )
            return PropsClass(**decoded_values)

    def props_to_index(
        self,
        props: Union[Dict[str, int], npt.NDArray[np.integer], None] = None,
        **kwargs: int
    ) -> Union[int, npt.NDArray[np.integer]]:
        """
        Convert property values to linear index.

        Supports partial property specifications: if only a subset of properties
        is provided, returns array of all indices matching those properties.

        Parameters
        ----------
        props : dict or ndarray of int or None, optional
            Property values to convert. Can be:
            - dict: mapping from property names to values (full or partial)
            - ndarray: property values in same order as self.properties (must be complete)
            - None: use kwargs instead
        **kwargs : int
            Alternative to dict: pass properties as keyword arguments (full or partial).

        Returns
        -------
        int or ndarray of int
            - If all properties specified: scalar index
            - If partial properties specified: array of matching indices
            - If props is 2D array: array of indices (one per row)

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
        >>> # Full specification -> scalar index
        >>> space.props_to_index({'a': 2, 'b': 1})
        5
        >>> space.props_to_index(a=2, b=1)  # kwargs alternative
        5
        >>> # Partial specification -> array of matching indices
        >>> space.props_to_index({'a': 2})  # All indices where a=2
        array([2, 5, 8])  # Corresponds to a=2,b=0; a=2,b=1; a=2,b=2
        >>> space.props_to_index(b=1)  # All indices where b=1
        array([1, 4, 7])  # Corresponds to a=0,b=1; a=1,b=1; a=2,b=1
        """
        # Handle kwargs
        if kwargs:
            if props is not None:
                raise ValueError("Cannot specify both props dict and kwargs")
            props = kwargs

        if props is None:
            raise ValueError("Must provide either props dict/array or kwargs")

        # Handle array input (must be complete specification)
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
                return np.dot(encoded, self._radix_powers)
            return int(np.dot(encoded, self._radix_powers))

        # Dict input - check if partial or complete
        prop_names = set(p.name for p in self.properties)
        specified_props = set(props.keys())

        # Validate that all specified properties are valid
        invalid_props = specified_props - prop_names
        if invalid_props:
            raise ValueError(f"Unknown properties: {invalid_props}")

        # Validate property values
        for prop_name, prop_val in props.items():
            prop = self.property_dict[prop_name]
            prop.validate_value(prop_val)

        # If all properties specified, return single index
        if specified_props == prop_names:
            encoded = np.array([
                prop.encode_value(props[prop.name])
                for prop in self.properties
            ])
            return int(np.dot(encoded, self._radix_powers))

        # Partial specification - generate all matching indices
        import itertools

        # Identify unspecified properties and their value ranges
        unspecified = []
        for prop in self.properties:
            if prop.name not in props:
                # Generate all valid values for this property
                values = list(range(prop.min_value, prop.max_value + 1))
                unspecified.append((prop.name, values))

        # Generate all combinations of unspecified properties
        if not unspecified:
            # All properties specified (shouldn't reach here, but handle it)
            encoded = np.array([
                prop.encode_value(props[prop.name])
                for prop in self.properties
            ])
            return int(np.dot(encoded, self._radix_powers))

        # Build list of all matching indices
        matching_indices = []

        # Get all combinations of unspecified property values
        unspec_names = [name for name, _ in unspecified]
        unspec_value_lists = [values for _, values in unspecified]

        for value_combo in itertools.product(*unspec_value_lists):
            # Merge specified and current unspecified values
            full_props = props.copy()
            for name, value in zip(unspec_names, value_combo):
                full_props[name] = value

            # Compute index for this complete property set
            encoded = np.array([
                prop.encode_value(full_props[prop.name])
                for prop in self.properties
            ])
            idx = int(np.dot(encoded, self._radix_powers))
            matching_indices.append(idx)

        matching_indices = np.array(sorted(matching_indices), dtype=int)

        if np.any(matching_indices >= self.n_states):
            raise IndexError(f"One or more computed indices out of range for property set of size {self.n_states}")

        return matching_indices


class PropertyDict(dict):
    """
    Dict subclass that provides attribute access to keys.

    Used to wrap property dictionaries for convenient attribute-style access.

    Examples
    --------
    >>> props = PropertyDict({'descendants': 5, 'population': 1})
    >>> props.descendants  # Instead of props['descendants']
    5
    >>> props.population = 2
    >>> props['population']
    2
    """

    def __getattr__(self, name: str):
        """Access dict keys as attributes."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No property named '{name}'")

    def __setattr__(self, name: str, value):
        """Set dict keys via attribute assignment."""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __dir__(self):
        """Include dict keys in dir() output for autocomplete."""
        return list(super().__dir__()) + list(self.keys())


class StateIndexer:
    """
    Manages multiple independent property sets with attribute access.

    StateIndexer allows organizing state variables into named PropertySets,
    where each PropertySet defines an independent combinatorial space. This
    is useful for separating primary state features (e.g., lineage properties)
    from auxiliary metadata (e.g., timing, identifiers).

    Parameters
    ----------
    property_sets : List[PropertySet] or List[Property], optional
        Either a list of PropertySet objects or a list of Property objects
        (which creates a single 'default' PropertySet for backward compatibility)
    **named_property_lists : dict
        Keyword arguments mapping PropertySet names to lists of Property objects

    Attributes
    ----------
    n_states : int
        Total states in first/default PropertySet (backward compatibility)

    Examples
    --------
    >>> # Create with keyword arguments
    >>> indexer = StateIndexer(
    ...     lineage=[Property('descendants', max_value=10)],
    ...     metadata=[Property('time_bin', max_value=1000)]
    ... )
    >>> indexer.lineage.n_states
    11
    >>> indexer.metadata.n_states
    1001

    >>> # Access via attributes
    >>> props = indexer.lineage.index_to_props(5)
    >>> props
    {'descendants': 5}

    >>> # Backward compatible with single PropertySet
    >>> indexer = StateIndexer([Property('descendants', max_value=10)])
    >>> indexer.n_states  # Delegates to default PropertySet
    11
    >>> indexer.index_to_props(5)  # Delegates to default PropertySet
    {'descendants': 5}
    """

    def __init__(
        self,
        property_sets: Optional[Union[List[PropertySet], List[Property]]] = None,
        **named_property_lists
    ):
        """
        Create StateIndexer with named property sets.

        Parameters
        ----------
        property_sets : List[PropertySet] or List[Property], optional
            Either a list of PropertySet objects or a list of Property objects
        **named_property_lists : dict
            Keyword arguments mapping names to Property lists

        Raises
        ------
        ValueError
            If no PropertySets provided or duplicate names detected

        Examples
        --------
        >>> # From PropertySet objects
        >>> pset1 = PropertySet('lineage', [Property('descendants', max_value=10)])
        >>> indexer = StateIndexer([pset1])

        >>> # From keyword args
        >>> indexer = StateIndexer(
        ...     lineage=[Property('descendants', max_value=10)],
        ...     metadata=[Property('time_bin', max_value=1000)]
        ... )

        >>> # Backward compatible
        >>> indexer = StateIndexer([Property('descendants', max_value=10)])
        """
        # Use object.__setattr__ to bypass __setattr__ during init
        object.__setattr__(self, '_property_sets', {})

        # Handle positional argument
        if property_sets is not None:
            if property_sets and isinstance(property_sets[0], PropertySet):
                # List of PropertySet objects
                for pset in property_sets:
                    if pset.name in self._property_sets:
                        raise ValueError(f"Duplicate PropertySet name: {pset.name}")
                    self._property_sets[pset.name] = pset
            elif property_sets and isinstance(property_sets[0], Property):
                # Backward compatibility: List of Property -> 'default' PropertySet
                self._property_sets['default'] = PropertySet('default', property_sets)
            else:
                raise ValueError("property_sets must be list of PropertySet or Property objects")

        # Handle keyword arguments
        for name, props in named_property_lists.items():
            if name in self._property_sets:
                raise ValueError(f"Duplicate PropertySet name: {name}")
            self._property_sets[name] = PropertySet(name, props)

        if not self._property_sets:
            raise ValueError("Must provide at least one PropertySet")

    def __getitem__(self, name: str) -> PropertySet:
        """Access PropertySet by name using indexer['name']."""
        return self._property_sets[name]

    def __getattr__(self, name: str) -> PropertySet:
        """Access PropertySet by name using indexer.name."""
        if name.startswith('_'):
            # Avoid recursion for private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._property_sets:
            return self._property_sets[name]
        raise AttributeError(f"No PropertySet named '{name}'")

    def __setattr__(self, name: str, value):
        """Prevent setting attributes that conflict with PropertySet names."""
        if hasattr(self, '_property_sets') and name in self._property_sets:
            raise AttributeError(f"Cannot modify PropertySet '{name}' via attribute assignment")
        object.__setattr__(self, name, value)

    def __dir__(self):
        """Include PropertySet names in dir() output for autocomplete."""
        return list(super().__dir__()) + list(self._property_sets.keys())

    def __contains__(self, name: str) -> bool:
        """Check if PropertySet exists."""
        return name in self._property_sets

    def keys(self):
        """Get PropertySet names."""
        return self._property_sets.keys()

    def values(self):
        """Get PropertySet objects."""
        return self._property_sets.values()

    def items(self):
        """Get (name, PropertySet) pairs."""
        return self._property_sets.items()

    def __len__(self):
        """Number of property sets."""
        return len(self._property_sets)

    # Backward compatibility - delegate to first/default PropertySet
    @property
    def n_states(self) -> int:
        """
        Size of first PropertySet (backward compatibility).

        For multi-PropertySet indexers, use indexer.name.n_states instead.
        """
        if len(self._property_sets) == 1:
            return next(iter(self._property_sets.values())).n_states
        elif 'default' in self._property_sets:
            return self._property_sets['default'].n_states
        else:
            raise ValueError("Use indexer.name.n_states for multi-set indexers")

    def index_to_props(self, index, as_dict=False, as_values=False):
        """
        Delegate to first PropertySet (backward compatibility).

        For multi-PropertySet indexers, use indexer.name.index_to_props() instead.

        Parameters
        ----------
        index : int or ndarray
            Index to convert
        as_dict : bool, default=False
            If True, return dict instead of dataclass
        as_values : bool, default=False
            If True, return array of values
        """
        if len(self._property_sets) == 1:
            return next(iter(self._property_sets.values())).index_to_props(index, as_dict, as_values)
        elif 'default' in self._property_sets:
            return self._property_sets['default'].index_to_props(index, as_dict, as_values)
        else:
            raise ValueError("Use indexer.name.index_to_props() for multi-set indexers")

    def props_to_index(self, props=None, **kwargs):
        """
        Delegate to first PropertySet (backward compatibility).

        For multi-PropertySet indexers, use indexer.name.props_to_index() instead.
        """
        if len(self._property_sets) == 1:
            return next(iter(self._property_sets.values())).props_to_index(props, **kwargs)
        elif 'default' in self._property_sets:
            return self._property_sets['default'].props_to_index(props, **kwargs)
        else:
            raise ValueError("Use indexer.name.props_to_index() for multi-set indexers")

    # Multi-set operations
    def state_to_indices(self, state: Dict[str, Dict[str, int]]) -> Dict[str, int]:
        """
        Convert full state to multi-index.

        Parameters
        ----------
        state : dict
            Nested dict {pset_name: {prop_name: value}}

        Returns
        -------
        dict
            Dict mapping PropertySet names to indices

        Examples
        --------
        >>> indexer = StateIndexer(
        ...     lineage=[Property('descendants', max_value=10)],
        ...     metadata=[Property('time_bin', max_value=1000)]
        ... )
        >>> state = {
        ...     'lineage': {'descendants': 5},
        ...     'metadata': {'time_bin': 100}
        ... }
        >>> indexer.state_to_indices(state)
        {'lineage': 5, 'metadata': 100}
        """
        return {
            name: self._property_sets[name].props_to_index(props)
            for name, props in state.items()
        }

    def indices_to_state(self, indices: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """
        Convert multi-index to full state.

        Parameters
        ----------
        indices : dict
            Dict mapping PropertySet names to indices

        Returns
        -------
        dict
            Nested dict {pset_name: {prop_name: value}}

        Examples
        --------
        >>> indexer = StateIndexer(
        ...     lineage=[Property('descendants', max_value=10)],
        ...     metadata=[Property('time_bin', max_value=1000)]
        ... )
        >>> indices = {'lineage': 5, 'metadata': 100}
        >>> indexer.indices_to_state(indices)
        {'lineage': {'descendants': 5}, 'metadata': {'time_bin': 100}}
        """
        return {
            name: self._property_sets[name].index_to_props(idx)
            for name, idx in indices.items()
        }

    def __repr__(self):
        """String representation of StateIndexer."""
        psets = ', '.join(f"{name}({ps.n_states})" for name, ps in self._property_sets.items())
        return f"StateIndexer({psets})"


class StateVector:
    """
    Dict-like interface for multi-PropertySet state with attribute access.

    StateVector wraps a StateIndexer and provides convenient access to properties
    via dictionary syntax or attributes. It maintains both indices and property
    dict representations for each PropertySet.

    Parameters
    ----------
    state_indexer : StateIndexer
        StateIndexer defining valid property sets
    indices : dict, optional
        Dict mapping PropertySet names to indices
    state : dict, optional
        Nested dict {pset_name: {prop_name: value}}

    Attributes
    ----------
    state_indexer : StateIndexer
        Associated state indexer
    indices : dict
        Current indices for each PropertySet
    state : dict
        Current property values for each PropertySet

    Examples
    --------
    >>> # Multi-PropertySet usage
    >>> indexer = StateIndexer(
    ...     lineage=[Property('descendants', max_value=10)],
    ...     metadata=[Property('time_bin', max_value=1000)]
    ... )
    >>> state_vec = StateVector(indexer, state={
    ...     'lineage': {'descendants': 5},
    ...     'metadata': {'time_bin': 100}
    ... })
    >>> state_vec.lineage.descendants  # Attribute access
    5
    >>> state_vec.metadata.time_bin
    100

    >>> # Backward compatible single PropertySet usage
    >>> indexer = StateIndexer([Property('descendants', max_value=10)])
    >>> state_vec = StateVector(indexer, indices={'default': 5})
    >>> state_vec['default']['descendants']
    5
    """

    def __init__(
        self,
        state_indexer: StateIndexer,
        indices: Optional[Dict[str, int]] = None,
        state: Optional[Dict[str, Dict[str, int]]] = None
    ) -> None:
        """
        Initialize state vector from indices or state.

        Parameters
        ----------
        state_indexer : StateIndexer
            StateIndexer defining valid property sets.
        indices : dict, optional
            Dict mapping PropertySet names to indices.
        state : dict, optional
            Nested dict {pset_name: {prop_name: value}}.

        Raises
        ------
        ValueError
            If both indices and state are specified, or neither is specified.
        """
        object.__setattr__(self, 'state_indexer', state_indexer)

        if indices is not None and state is not None:
            raise ValueError("Cannot specify both indices and state")

        if indices is not None:
            object.__setattr__(self, '_indices', indices)
            object.__setattr__(self, '_state', {
                name: PropertyDict(props)
                for name, props in state_indexer.indices_to_state(indices).items()
            })
        elif state is not None:
            # Convert to PropertyDict for attribute access
            object.__setattr__(self, '_state', {
                name: PropertyDict(props) if not isinstance(props, PropertyDict) else props
                for name, props in state.items()
            })
            object.__setattr__(self, '_indices', state_indexer.state_to_indices(state))
        else:
            raise ValueError("Must specify either indices or state")

    def __getitem__(self, key: Union[str, tuple]):
        """
        Access using indexer syntax: state['lineage'] or state['lineage', 'descendants'].

        Parameters
        ----------
        key : str or tuple
            Either PropertySet name or (PropertySet name, property name)

        Returns
        -------
        PropertyDict or int
            PropertyDict if key is PropertySet name, int if specific property
        """
        if isinstance(key, tuple):
            pset_name, prop_name = key
            return self._state[pset_name][prop_name]
        else:
            return self._state[key]

    def __getattr__(self, name: str):
        """
        Access using attribute syntax: state.lineage or state.descendants.

        For PropertySet names, returns the PropertyDict.
        For property names, searches all PropertySets (must be unique).

        Parameters
        ----------
        name : str
            PropertySet or property name

        Returns
        -------
        PropertyDict or int
            PropertyDict if PropertySet name, int if property name

        Raises
        ------
        AttributeError
            If name not found or is ambiguous
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if it's a PropertySet name
        if name in self._state:
            return self._state[name]

        # Search for property name across all PropertySets
        found_value = None
        found_pset = None
        for pset_name, props in self._state.items():
            if name in props:
                if found_value is not None:
                    raise AttributeError(
                        f"Property '{name}' is ambiguous (found in '{found_pset}' and '{pset_name}'). "
                        f"Use state.{found_pset}.{name} or state.{pset_name}.{name}"
                    )
                found_value = props[name]
                found_pset = pset_name

        if found_value is not None:
            return found_value

        raise AttributeError(f"No PropertySet or property named '{name}'")

    def __setattr__(self, name: str, value):
        """Prevent direct attribute assignment (use update methods instead)."""
        if name.startswith('_') or name in ('state_indexer',):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Cannot set '{name}' directly. "
                f"Use state.lineage.prop_name = value and call update_indices()"
            )

    def __dir__(self):
        """Include PropertySet and property names for autocomplete."""
        base_attrs = list(super().__dir__())
        pset_names = list(self._state.keys())

        # Add property names if unique across PropertySets
        prop_counts = {}
        for props in self._state.values():
            for prop_name in props.keys():
                prop_counts[prop_name] = prop_counts.get(prop_name, 0) + 1
        unique_props = [name for name, count in prop_counts.items() if count == 1]

        return base_attrs + pset_names + unique_props

    @property
    def indices(self) -> Dict[str, int]:
        """
        Get current indices.

        Returns
        -------
        dict
            Copy of indices dict mapping PropertySet names to indices
        """
        return self._indices.copy()

    @property
    def state(self) -> Dict[str, Dict[str, int]]:
        """
        Get current full state.

        Returns
        -------
        dict
            Nested dict {pset_name: {prop_name: value}}
        """
        return {k: dict(v) for k, v in self._state.items()}

    def update_indices(self):
        """
        Recompute indices from current state.

        Call this after modifying property values via attribute/item assignment.
        """
        object.__setattr__(
            self,
            '_indices',
            self.state_indexer.state_to_indices(self._state)
        )

    def copy(self) -> 'StateVector':
        """
        Create a copy of this state vector.

        Returns
        -------
        StateVector
            New StateVector with same indexer and property values.
        """
        return StateVector(self.state_indexer, state=self.state)

    def __repr__(self) -> str:
        """
        String representation of state vector.

        Returns
        -------
        str
            Human-readable representation showing indices
        """
        indices_str = ', '.join(f"{k}={v}" for k, v in self._indices.items())
        return f"StateVector({indices_str})"


# Backward compatibility alias
class StateSpace(PropertySet):
    """
    Backward compatibility alias for PropertySet.

    .. deprecated:: 0.23.0
        Use PropertySet instead. StateSpace will be removed in a future version.

    This class provides backward compatibility with code written for the old
    StateSpace API. New code should use PropertySet directly, or preferably
    use StateIndexer for managing multiple property sets.

    Examples
    --------
    >>> # Old API (still works)
    >>> space = StateSpace('default', [Property('descendants', max_value=10)])
    >>> space.size  # Deprecated attribute
    11

    >>> # New API (preferred)
    >>> pset = PropertySet('lineage', [Property('descendants', max_value=10)])
    >>> pset.n_states
    11
    """

    def __init__(self, properties: List[Property], name: str = 'default') -> None:
        """
        Initialize StateSpace (backward compatibility).

        Parameters
        ----------
        properties : list of Property
            List of properties defining the state space
        name : str, optional
            Name of this property set (default: 'default')

        Notes
        -----
        This constructor signature is deprecated. Use PropertySet(name, properties) instead.
        """
        import warnings
        warnings.warn(
            "StateSpace is deprecated. Use PropertySet(name, properties) or StateIndexer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Call PropertySet with swapped argument order for backward compatibility
        super().__init__(name, properties)

    @property
    def size(self) -> int:
        """
        Total number of states (backward compatibility).

        .. deprecated:: 0.23.0
            Use n_states instead.
        """
        import warnings
        warnings.warn(
            "StateSpace.size is deprecated. Use PropertySet.n_states instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.n_states
