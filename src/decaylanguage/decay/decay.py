# Copyright (c) 2018-2026, Eduardo Rodrigues and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/decaylanguage for details.


from __future__ import annotations

import typing
from collections import Counter
from collections.abc import Collection, Iterator, Sequence
from copy import deepcopy
from itertools import product
from typing import Any, NoReturn, TypedDict

from particle import PDGID, ParticleNotFound
from particle.converters import EvtGenName2PDGIDBiMap
from particle.exceptions import MatchingIDNotFound

from .._compat.typing import Self
from ..utils import DescriptorFormat, charge_conjugate_name


class DecayModeDict(TypedDict):
    bf: float
    fs: Sequence[str | DecayChainDict]
    model: str
    model_params: str | Sequence[str | Any]


DecayChainDict = dict[str, list[DecayModeDict]]


class DaughtersDict(Counter[str]):
    """
    Class holding a decay final state as a dictionary.

    It is a building block for the digital representation of full decay chains.
    A final state such as 'K+ K- K- pi+ pi0' is stored as
    ``{'K+': 1, 'K-': 2, 'pi+': 1, 'pi0': 1}``.

    Parameters
    ----------
    iterable : dict, collection of str or str, optional
        The final-state particles, given as a dictionary of particle names
        and multiplicities, as a collection of particle names, or as a single
        string of space-separated particle names. Entries of a dictionary
        with a non-positive multiplicity are dropped. Defaults to an empty
        final state.
    **kwds
        Extra particle multiplicities given as keyword arguments,
        for example ``gamma=2``.

    Notes
    -----
    This class assumes EvtGen particle names, though this assumption is only relevant
    for the ``charge_conjugate`` method (refer to its documentation).
    Otherwise, all other class methods smoothly deal with
    any kind of particle names (basically an iterable of strings).

    Examples
    --------
    >>> # An empty final state
    >>> dd = DaughtersDict()

    >>> # Constructor from a dictionary
    >>> dd = DaughtersDict({'K+': 1, 'K-': 2, 'pi+': 1, 'pi0': 3})

    >>> # Constructor from a list of particle names
    >>> dd = DaughtersDict(['K+', 'K-', 'K-', 'pi+', 'pi0'])

    >>> # Constructor from a string representing the final state
    >>> dd = DaughtersDict('K+ K- pi0')

    >>> # "Mixed" constructor
    >>> dd = DaughtersDict('K+ K-', pi0=1, gamma=2)
    """

    def __init__(
        self,
        iterable: dict[str, int] | Collection[str] | str | None = None,
        **kwds: Any,
    ) -> None:
        if isinstance(iterable, dict):
            iterable = {k: v for k, v in iterable.items() if v > 0}
        elif iterable and isinstance(iterable, str):
            iterable = iterable.split()
        super().__init__(iterable, **kwds)

    @classmethod
    def fromkeys(cls, iterable: Any, v: Any = None) -> NoReturn:
        """
        Not supported, as for :class:`collections.Counter`.

        Parameters
        ----------
        iterable : Any
            Ignored.
        v : Any, optional
            Ignored.

        Raises
        ------
        NotImplementedError
            Always. Use ``DaughtersDict(iterable)`` instead.
        """
        # ==> Comment copied from Counter.fromkeys():
        # There is no equivalent method for counters because the semantics
        # would be ambiguous in cases such as Counter.fromkeys('aaabbc', v=2).
        # Initializing counters to zero values isn't necessary because zero
        # is already the default value for counter lookups.  Initializing
        # to one is easily accomplished with Counter(set(iterable)).  For
        # more exotic cases, create a dictionary first using a dictionary
        # comprehension or dict.fromkeys().
        raise NotImplementedError(
            "DaughtersDict.fromkeys() is undefined, just as Counter.fromkeys(). Use DaughtersDict(iterable) instead."
        )

    def to_string(self) -> str:
        """
        Return the daughters as a string representation (ordered list of names).

        Returns
        -------
        str
            The sorted particle names, separated by spaces.

        Examples
        --------
        >>> DaughtersDict({'pi0': 1, 'K-': 2}).to_string()
        'K- K- pi0'
        """
        return " ".join(sorted(self.elements()))

    def to_list(self) -> list[str]:
        """
        Return the daughters as an ordered list of names.

        Returns
        -------
        list of str
            The sorted particle names, each repeated by its multiplicity.

        Examples
        --------
        >>> DaughtersDict({'pi0': 1, 'K-': 2}).to_list()
        ['K-', 'K-', 'pi0']
        """
        return sorted(self.elements())

    def charge_conjugate(self, pdg_name: bool = False) -> Self:
        """
        Return the charge-conjugate final state.

        Parameters
        ----------
        pdg_name : bool, optional
            Input particle names are PDG names,
            not the (default) EvtGen names. Default is False.

        Returns
        -------
        DaughtersDict
            A new instance with every particle replaced by its
            charge conjugate.

        Examples
        --------
        >>> dd = DaughtersDict({'K+': 2, 'pi0': 1})
        >>> dd.charge_conjugate()
        <DaughtersDict: ['K-', 'K-', 'pi0']>
        >>>
        >>> dd = DaughtersDict({'K_S0': 1, 'pi0': 1})
        >>> dd.charge_conjugate()
        <DaughtersDict: ['K_S0', 'pi0']>
        >>>
        >>> dd = DaughtersDict({'K(S)0': 1, 'pi+': 1})  # PDG names!
        >>> # 'K(S)0' unrecognised in charge conjugation unless specified that these are PDG names
        >>> dd.charge_conjugate()
        <DaughtersDict: ['ChargeConj(K(S)0)', 'pi-']>
        >>> dd.charge_conjugate(pdg_name=True)
        <DaughtersDict: ['K(S)0', 'pi-']>
        """
        return self.__class__(
            {charge_conjugate_name(p, pdg_name=pdg_name): n for p, n in self.items()}
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.to_list()}>"

    def __str__(self) -> str:
        return repr(self)

    def __len__(self) -> int:
        """
        Return the length, i.e. the number of final-state particles.

        Returns
        -------
        int
            The sum of all particle multiplicities.

        Notes
        -----
        This is generally *not* the number of dictionary elements.
        """
        return sum(self.values())

    def __add__(self, other: Self) -> Self:  # type: ignore[override]
        """
        Add two final states, particle-type-wise.

        Parameters
        ----------
        other : DaughtersDict
            The final state to add.

        Returns
        -------
        DaughtersDict
            A new instance with the multiplicities of both final states summed.

        Examples
        --------
        >>> dd1 = DaughtersDict({'K+': 1, 'K-': 1})
        >>> dd2 = DaughtersDict({'pi0': 1})
        >>> dd1 + dd2
        <DaughtersDict: ['K+', 'K-', 'pi0']>
        """
        dd = super().__add__(other)
        return self.__class__(dd)

    def __sub__(self, other: Self) -> Self:  # type: ignore[override]
        """
        Subtract two final states, particle-type-wise.

        Parameters
        ----------
        other : DaughtersDict
            The final state to subtract.

        Returns
        -------
        DaughtersDict
            A new instance with the multiplicities subtracted. Particles whose
            multiplicity drops to zero or below are removed.

        Examples
        --------
        >>> dd1 = DaughtersDict({'K+': 1, 'K-': 2, 'pi0': 3})
        >>> dd2 = DaughtersDict({'K-': 1, 'pi0': 1})
        >>> dd1 - dd2
        <DaughtersDict: ['K+', 'K-', 'pi0', 'pi0']>
        """
        dd = super().__sub__(other)
        return self.__class__(dd)

    def __iter__(self) -> Iterator[str]:
        return self.elements()


class DecayMode:
    """
    Class holding a particle decay mode.

    A decay mode is typically a branching fraction and a list of final-state
    particles (i.e. a list of DaughtersDict instances).
    The class can also contain metadata such as decay model and optional
    decay-model parameters, as defined for example in .dec decay files.

    This class is a building block for the digital representation
    of full decay chains.

    Parameters
    ----------
    bf : float, optional
        Decay mode branching fraction. Default is 0.
    daughters : DaughtersDict, dict, list, tuple or str, optional
        The final-state particles, in any form accepted by ``DaughtersDict``.
        Default is None, meaning an empty final state, unless an ``fs``
        keyword argument is given, in which case it is used instead.
    **info
        Decay mode model information and/or user metadata (aka extra info).
        By default the following elements are always created:
        dict(model='', model_params='').
        The user can provide any metadata, see the examples below.

    Notes
    -----
    This class assumes EvtGen particle names, though this assumption is only
    relevant for the ``charge_conjugate`` method.
    Otherwise, all other class methods smoothly deal
    with any kind of particle names (basically an iterable of strings).

    Examples
    --------
    >>> # A 'default' and hence empty, decay mode
    >>> dm = DecayMode()

    >>> # Decay mode with minimal input information
    >>> dd = DaughtersDict('K+ K-')
    >>> dm = DecayMode(0.5, dd)

    >>> # Decay mode with minimal input information, simpler version
    >>> dm = DecayMode(0.5, 'K+ K-')

    >>> # Decay mode with decay model information
    >>> dd = DaughtersDict('pi- pi0 nu_tau')
    >>> dm = DecayMode(0.2551, dd,
    ...                model='TAUHADNU',
    ...                model_params=[-0.108, 0.775, 0.149, 1.364, 0.400])

    >>> # Decay mode with user metadata
    >>> dd = DaughtersDict('K+ K-')
    >>> dm = DecayMode(0.5, dd, model='PHSP', study='toy', year=2019)

    >>> # Decay mode with metadata for generators such as zfit's phasespace
    >>> dm = DecayMode(0.5, "K+ K-", zfit={"B0": "gauss"})
    >>> dm.metadata['zfit'] == {'B0': 'gauss'}
    True
    """

    __slots__ = ("bf", "daughters", "metadata")

    def __init__(
        self,
        bf: float = 0,
        daughters: (
            DaughtersDict | dict[str, int] | list[str] | tuple[str, ...] | str | None
        ) = None,
        **info: Any,
    ) -> None:
        self.bf = bf
        if daughters is None and "fs" in info:
            daughters = info.pop("fs")
        self.daughters = DaughtersDict(daughters)

        self.metadata: dict[str, Any] = {"model": "", "model_params": ""}
        self.metadata.update(**info)

    @classmethod
    def from_dict(
        cls,
        decay_mode_dict: DecayModeDict,
    ) -> Self:
        """
        Constructor from a dictionary of the form {'bf': <float>, 'fs': [...], ...}.

        Parameters
        ----------
        decay_mode_dict : dict
            The decay mode. The keys 'bf' and 'fs' are mandatory. All others
            are interpreted as model information or metadata, see the class
            signature and doc. The input is not modified.

        Returns
        -------
        DecayMode
            The decay mode built from the dictionary.

        Raises
        ------
        RuntimeError
            If ``decay_mode_dict`` lacks the 'bf' or the 'fs' key.

        Notes
        -----
        This class assumes EvtGen particle names, though this assumption is only
        relevant for the ``charge_conjugate`` method.
        Otherwise, all other class methods smoothly deal
        with any kind of particle names (basically an iterable of strings).

        Examples
        --------
        >>> # Simplest construction
        >>> DecayMode.from_dict({'bf': 0.98823, 'fs': ['gamma', 'gamma']})
        <DecayMode: daughters=gamma gamma, BF=0.98823>

        >>> # Decay mode with decay model details
        >>> DecayMode.from_dict({'bf': 0.98823,
        ...                      'fs': ['gamma', 'gamma'],
        ...                      'model': 'PHSP',
        ...                      'model_params': ''})
        <DecayMode: daughters=gamma gamma, BF=0.98823>

        >>> # Decay mode with metadata for generators such as zfit's phasespace
        >>> dm = DecayMode.from_dict({'bf': 0.5, 'fs': ["K+, K-"], "zfit": {"B0": "gauss"}})
        >>> dm.metadata
        {'model': '', 'model_params': '', 'zfit': {'B0': 'gauss'}}
        """
        dm = deepcopy(decay_mode_dict)

        # Ensure the input dict has the 2 required keys 'bf' and 'fs'
        if not dm.keys() >= {"bf", "fs"}:
            raise RuntimeError("Input not in the expected format! Needs 'bf' and 'fs'")

        return cls(**dm)

    @classmethod
    def from_pdgids(
        cls,
        bf: float = 0,
        daughters: list[int] | tuple[int, ...] | None = None,
        **info: Any,
    ) -> Self:
        """
        Constructor for a final state given as a list of particle PDG IDs.

        Parameters
        ----------
        bf : float, optional
            Decay mode branching fraction. Default is 0.
        daughters : list or tuple of int, optional
            The final-state particle PDG IDs. Default is None,
            meaning an empty final state.
        **info
            Decay mode model information and/or user metadata (aka extra info).
            By default the following elements are always created:
            dict(model='', model_params='').
            The user can provide any metadata, see the examples below.

        Returns
        -------
        DecayMode
            The decay mode, with the final-state particles stored
            under their EvtGen names.

        Raises
        ------
        particle.ParticleNotFound
            If a PDG ID has no matching EvtGen name.

        Notes
        -----
        All particle names are internally saved as EvtGen names,
        to be consistent with the default class assumption, see class docstring.

        Examples
        --------
        >>> DecayMode.from_pdgids(0.5, [321, -321])
        <DecayMode: daughters=K+ K-, BF=0.5>

        >>> DecayMode.from_pdgids(0.5, (310, 310))
        <DecayMode: daughters=K_S0 K_S0, BF=0.5>

        >>> # Decay mode with metadata
        >>> dm = DecayMode.from_pdgids(0.5, (310, 310), model="PHSP")
        >>> dm.metadata
        {'model': 'PHSP', 'model_params': ''}
        """
        if not daughters:
            return cls(bf=bf, daughters=None, **info)

        try:
            _daughters = [EvtGenName2PDGIDBiMap[PDGID(d)] for d in daughters]
        except MatchingIDNotFound:
            # The bi-map raises a MatchingIDNotFound for missed match
            # but better and more natural to raise a "particle not found"
            # as the former is a technical, implementation, detail.
            raise ParticleNotFound("Please check your input PDG IDs!") from None

        # Override the default settings with the user input, if any
        return cls(bf=bf, daughters=_daughters, **info)

    def describe(self) -> str:
        """
        Make a nice high-density string for all decay-mode properties and info.

        Returns
        -------
        str
            A multi-line description with the daughters, the branching fraction,
            the decay model and its parameters, and any extra info.

        Examples
        --------
        >>> print(DecayMode(0.5, 'K+ K-', model='PHSP').describe())  # doctest: +NORMALIZE_WHITESPACE
        Daughters: K+ K- , BF: 0.5
            Decay model: PHSP
        """
        _daughters = " ".join(self.daughters)
        _model = self.metadata["model"]
        _model_params = (
            self.metadata["model_params"]
            if self.metadata["model_params"] is not None
            else ""
        )
        val = (
            f"Daughters: {_daughters} , BF: {self.bf:<15.8g}\n"
            f"    Decay model: {_model} {_model_params}"
        )

        keys = [k for k in self.metadata if k not in ("model", "model_params")]
        if keys:
            val += "\n    Extra info:\n"
        for key in keys:
            val += f"        {key}: {self.metadata[key]}\n"

        return val

    def to_dict(self) -> dict[str, int | float | str | list[str]]:
        """
        Return the decay mode as a dictionary.

        The format is the one understood by the ``DecayChainViewer`` class.

        Returns
        -------
        dict
            The keys 'bf' and 'fs' (sorted list of daughter names),
            followed by the decay model information and any metadata.

        Examples
        --------
        >>> dm = DecayMode(0.5, 'K+ K- K- pi- pi0 nu_tau', model='PHSP', study='toy', year=2019)
        >>> dm.to_dict()    # doctest: +NORMALIZE_WHITESPACE
        {'bf': 0.5,
         'fs': ['K+', 'K-', 'K-', 'nu_tau', 'pi-', 'pi0'],
         'model': 'PHSP',
         'model_params': '',
         'study': 'toy',
         'year': 2019}
        """
        d = {"bf": self.bf, "fs": self.daughters.to_list()}
        d.update(self.metadata)
        if d["model_params"] is None:
            d["model_params"] = ""
        return d  # type: ignore[return-value]

    def charge_conjugate(self, pdg_name: bool = False) -> Self:
        """
        Return the charge-conjugate decay mode.

        Parameters
        ----------
        pdg_name : bool, optional
            Input particle names are PDG names,
            not the (default) EvtGen names. Default is False.

        Returns
        -------
        DecayMode
            A new decay mode with the same branching fraction and metadata,
            and the charge-conjugate final state.

        Examples
        --------
        >>> dm = DecayMode(1.0, 'K+ K+ pi-')
        >>> dm.charge_conjugate()
        <DecayMode: daughters=K- K- pi+, BF=1.0>
        >>>
        >>> dm = DecayMode(1.0, 'K_S0 pi+')
        >>> dm.charge_conjugate()
        <DecayMode: daughters=K_S0 pi-, BF=1.0>
        >>>
        >>> dm = DecayMode(1.0, 'K(S)0 pi+')  # PDG names!
        >>> dm.charge_conjugate(pdg_name=True)
        <DecayMode: daughters=K(S)0 pi-, BF=1.0>
        """
        return self.__class__(
            self.bf, self.daughters.charge_conjugate(pdg_name), **self.metadata
        )

    def __len__(self) -> int:
        """
        Return the "length" of a decay mode, i.e. the number of final-state particles.

        Returns
        -------
        int
            The number of final-state particles.
        """
        return len(self.daughters)

    def __repr__(self) -> str:
        daughters = self.daughters.to_string() if len(self.daughters) > 0 else "[]"
        return f"<{self.__class__.__name__}: daughters={daughters}, BF={self.bf}>"

    def __str__(self) -> str:
        return repr(self)


def _has_no_subdecay(ds: list[Any]) -> bool:
    """
    Check whether the input list is an end-of-chain final state.

    Internal function. The alternative is a final state with further sub-decays.

    Parameters
    ----------
    ds : list
        A final state, as found under the 'fs' key of a decay mode dictionary.

    Returns
    -------
    bool
        True if all elements are particle names (no sub-decays), False otherwise.

    Examples
    --------
    >>> # End-of-chain final state
    >>> _has_no_subdecay(['pi+', 'pi-'])
    True
    >>> # Final state with further sub-decays
    >>> _has_no_subdecay(
    ...     [{'K_S0': [{'bf': 0.692, 'fs': ['pi+', 'pi-'], 'model': '', 'model_params': ''}]},
    ...      'pi+',
    ...      'pi-'])
    False
    """
    return all(isinstance(p, str) for p in ds)


def _build_decay_modes(
    decay_modes: dict[str, DecayMode], dc_dict: DecayChainDict
) -> None:
    """
    Build all ``DecayMode`` instances contained in a decay chain dictionary.

    Internal recursive function. The input is the dict representation of
    a ``DecayChain``, which is for example the format returned by
    ``DecFileParser.build_decay_chains(...)``. The ``decay_modes`` dict
    is filled in place with mother particles and their final states
    as ``DecayMode`` instances.

    Parameters
    ----------
    decay_modes : dict
        A dict to be populated with the decay modes ``DecayMode``
        built from the input decay chain dictionary.
    dc_dict : dict
        The input decay chain dictionary.

    Raises
    ------
    RuntimeError
        If a decay mode has no 'fs' key, or if a particle is given
        two different decay modes (not a single decay chain).

    Notes
    -----
    Only single chains are supported, meaning every decaying particle
    can only define a single decay mode.

    Examples
    --------
    The simple example with no sub-decays:
    >>> dc_dict = {
    ...     "anti-D0": [
    ...         {
    ...           "bf": 1.0,
    ...           "fs": ["K+", "pi-"],
    ...           "model": "PHSP",
    ...           "model_params": ""
    ...         }
    ...     ]
    ... }
    >>> # It provides
    >>> decay_modes = {}
    >>> _build_decay_modes(decay_modes, dc_dict)
    >>> decay_modes
    {'anti-D0': <DecayMode: daughters=K+ pi-, BF=1.0>}

    A more complicated example with a sub-decay and more than one mode
    {
        "anti-D*0": [
            {
                "bf": 0.619,
                "fs": [
                    {
                        "anti-D0": [
                            {
                                "bf": 1.0,
                                "fs": ["K+", "pi-"],
                                "model": "PHSP",
                                "model_params": ""
                            }
                        ]
                    },
                    "pi0"
                ],
                "model": "VSS",
                "model_params": ""
            }
        ]
    }
    provides
    {'anti-D0': <DecayMode: daughters=K+ pi-, BF=1.0>,
     'anti-D*0': <DecayMode: daughters=anti-D0 pi0, BF=0.619>}
    """
    mother = next(iter(dc_dict.keys()))
    dms = dc_dict[mother]

    def _set_decay_mode(mother: str, decay_mode: DecayMode) -> None:
        # Single decay chains are allowed, which means a particle
        # cannot have 2 *different* decay modes. The same particle may, however,
        # appear several times in a chain with the same sub-decay.
        # For example, eta -> pi0 pi0 with pi0 -> gamma gamma, in which case
        # the identical mode is simply re-confirmed rather than rejected.
        existing = decay_modes.get(mother)
        if existing is not None and existing.to_dict() != decay_mode.to_dict():
            raise RuntimeError("Input is not a single decay chain!")
        decay_modes[mother] = decay_mode

    for dm in dms:
        try:
            fs = dm["fs"]
        except Exception as e:
            raise RuntimeError(
                "Internal dict representation not in the expected format - no 'fs' key is present!"
            ) from e

        assert isinstance(fs, list)
        if _has_no_subdecay(fs):
            _set_decay_mode(mother, DecayMode.from_dict(dm))
        else:
            d = deepcopy(dm)
            fs_local = d["fs"]
            assert isinstance(fs_local, list)
            for i, ifs in enumerate(fs_local):
                if isinstance(ifs, dict):
                    # Replace the element with the key and
                    # store the present decay mode ignoring sub-decays
                    fs_local[i] = next(iter(ifs.keys()))
                    # Recursively continue ...
                    _build_decay_modes(decay_modes, fs[i])
            # Create the decay mode now that none of its particles
            # has a sub-decay
            _set_decay_mode(mother, DecayMode.from_dict(d))


T = typing.TypeVar("T")


def _get_modes(decay_chain: dict[str, T]) -> T:
    # The list of decay modes is the first (and only) value of the dict
    assert len(decay_chain.values()) == 1
    modes = list(decay_chain.values())
    return modes[0]


def _get_fs(decay: DecayModeDict) -> list[Any]:
    """
    Return the final state of a decay mode dictionary.

    Parameters
    ----------
    decay : dict
        A decay mode dictionary.

    Returns
    -------
    list
        The value of the 'fs' key.

    Raises
    ------
    TypeError
        If the 'fs' value is not a list.

    Notes
    -----
    Helper function not meant to be used directly!
    """
    fs = decay["fs"]
    if isinstance(fs, list):
        return fs
    # This line should never be reached given that _get_fs is only used inside the
    # internal helper function _expand_decay_modes,
    # which by construction contains a valid decay chain dict
    raise TypeError(f"Expected list, not {type(fs)}")  # pragma: no cover


def _expand_decay_modes(
    decay_chain: DecayChainDict,
    *,
    top: bool = True,
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """
    Recursively replace all decay modes of a decay chain with decay descriptors.

    The input is a dict with a single key -- the mother particle -- whose value
    is a list of decay modes. The input ``decay_chain`` dict is updated in-place.

    Parameters
    ----------
    decay_chain : dict
        A dict representing decay chains, such as returned by DecayChain.to_dict
        or DecFileParser.build_decay_chains.
    top : bool, optional
        Whether the passed decay chain is the top-level or not (should usually
        be True: only really set to False when recursing the function).
        Default is True.
    aliases : dict of str to str, optional
        Mapping of names to replace. Useful when dealing with DecFiles
        that have 'Alias' statements. Default is None.

    Returns
    -------
    list of str
        The decay descriptors, which are also stored in ``decay_chain``
        under the (original) mother name.

    Notes
    -----
    Helper function not meant to be used directly!

    Examples
    --------
    A simple example with no sub-decays:
    {
        "anti-D0": [
            {
              "bf": 1.0,
              "fs": ["K+", "pi-"],
              "model": "PHSP",
              "model_params": ""
            }
        ]
    }
    becomes the ``decay_chain`` dict
    {
        "anti-D0": [
            "anti-D0 -> K+ pi-"
        ]
    }

    A more complicated example with a sub-decay and more than one mode:
    {
        "anti-D*0": [
            {
                "bf": 0.619,
                "fs": [
                    {
                        "anti-D0": [
                            {
                                "bf": 1.0,
                                "fs": ["K+", "pi-"],
                                "model": "PHSP",
                                "model_params": ""
                            }
                        ]
                    },
                    "pi0"
                ],
                "model": "VSS",
                "model_params": ""
            },
            {
                "bf": 0.381,
                "fs": [
                    {
                        "anti-D0": [
                            {
                                "bf": 1.0,
                                "fs": ["K+", "pi-"],
                                "model": "PHSP",
                                "model_params": ""
                            }
                        ]
                    },
                    "gamma"
                ],
                "model": "VSP_PWAVE",
                "model_params": ""
            }
        ]
    }
    becomes the ``decay_chain`` dict
    {
        "anti-D*0": [
            "anti-D*0 -> (anti-D0 -> K+ pi-) pi0",
            "anti-D*0 -> (anti-D0 -> K+ pi-) gamma",
        ]
    }

    And an example alias dict:
    {"MyAntiD0": "anti-D0"}
    can be used with
    {
        "MyAntiD0": [
            {
              "bf": 1.0,
              "fs": ["K+", "pi-"],
              "model": "PHSP",
              "model_params": ""
            }
        ]
    }
    to result in
    {
        'anti-D0': [
            'anti-D0 -> K+ pi-'
        ]
    }
    """

    # The mother particle is the first (and only) key of the dict
    assert len(decay_chain.keys()) == 1
    orig_mother = next(iter(decay_chain.keys()))
    mother = aliases.get(orig_mother, orig_mother) if aliases else orig_mother

    for mode in _get_modes(decay_chain):
        for fsp in _get_fs(mode):
            if isinstance(fsp, dict):
                _expand_decay_modes(fsp, top=False, aliases=aliases)

    # Replace dicts with strings (decay descriptors)
    expanded_modes = []
    for mode in _get_modes(decay_chain):
        fsp_options: list[list[str]] = []
        for fsp in _get_fs(mode):
            if isinstance(fsp, dict):
                fsp_options.append(_get_modes(fsp))
            elif isinstance(fsp, str):
                fsp_options.append([aliases.get(fsp, fsp) if aliases else fsp])
        for expanded_mode in product(*fsp_options):
            # TODO: delegate descriptor-building to another function
            #       allow for different conventions?
            final_state = DaughtersDict(expanded_mode).to_string()
            descriptor = DescriptorFormat.format_descriptor(mother, final_state, top)
            expanded_modes += [descriptor]

    decay_chain[orig_mother] = expanded_modes  # type: ignore[assignment]

    return expanded_modes


class DecayChain:
    """
    Class holding a particle (single) decay chain.

    A decay chain is typically a top-level decay
    (mother particle, branching fraction and final-state particles)
    and a set of sub-decays for any non-stable particle in the top-level decay.
    The whole chain can be seen as a mother particle and a list of chained decay modes.

    This class is the main building block for the digital representation
    of full decay chains.

    Parameters
    ----------
    mother : str
        Input mother particle of the top-level decay.
    decays : dict of str to DecayMode
        The decay modes, keyed by the name of the decaying particle.
        Must include the mother particle.

    Raises
    ------
    RuntimeError
        If ``decays`` has no entry for ``mother``.

    Notes
    -----
    1) Only single chains are supported, meaning every decaying particle
    can only define a single decay mode.
    2) This class does not assume any kind of particle names (EvtGen, PDG).
    It is nevertheless advised to default use EvtGen names for consistency
    with the defaults used in the related classes ``DecayMode`` and ``DaughtersDict``,
    unless there is a good motivation not to.

    Examples
    --------
    >>> dm1 = DecayMode(0.0124, 'K_S0 pi0', model='PHSP')
    >>> dm2 = DecayMode(0.692, 'pi+ pi-')
    >>> dm3 = DecayMode(0.98823, 'gamma gamma')
    >>> dc = DecayChain('D0', {'D0':dm1, 'K_S0':dm2, 'pi0':dm3})
    """

    __slots__ = ("decays", "mother")

    def __init__(self, mother: str, decays: dict[str, DecayMode]) -> None:
        if mother not in decays:
            raise RuntimeError("Input decay modes do not include the mother particle!")

        self.mother = mother
        self.decays = decays

    @classmethod
    def from_dict(cls, decay_chain_dict: DecayChainDict) -> Self:
        """
        Constructor from a decay chain represented as a dictionary.

        Parameters
        ----------
        decay_chain_dict : dict
            The decay chain, with the mother particle as its single key.
            The format is the same as that returned by
            ``DecFileParser.build_decay_chains(...)``.

        Returns
        -------
        DecayChain
            The decay chain, with one ``DecayMode`` per decaying particle.

        Raises
        ------
        RuntimeError
            If ``decay_chain_dict`` does not have exactly one key, if a decay mode
            has no 'fs' key, or if a particle is given two different decay modes.

        Examples
        --------
        >>> dc_dict = {'D0': [{'bf': 0.028,
        ...                    'fs': [{'K_S0': [{'bf': 0.692, 'fs': ['pi+', 'pi-']}]},
        ...                           'pi+', 'pi-']}]}
        >>> DecayChain.from_dict(dc_dict)
        <DecayChain: D0 -> K_S0 pi+ pi- (1 sub-decays), BF=0.028>
        """
        try:
            assert len(decay_chain_dict.keys()) == 1
        except Exception as e:
            raise RuntimeError("Input not in the expected format!") from e

        decay_modes: dict[str, DecayMode] = {}
        mother = next(iter(decay_chain_dict.keys()))
        _build_decay_modes(decay_modes, decay_chain_dict)

        return cls(mother, decay_modes)

    def top_level_decay(self) -> DecayMode:
        """
        Return the top-level decay as a ``DecayMode`` instance.

        Returns
        -------
        DecayMode
            The decay mode of the mother particle.
        """
        return self.decays[self.mother]

    @property
    def bf(self) -> float:
        """
        Branching fraction of the top-level decay.

        Returns
        -------
        float
            The branching fraction of the mother particle's decay mode.
        """
        return self.top_level_decay().bf

    @property
    def visible_bf(self) -> float:
        """
        Visible branching fraction of the whole decay chain.

        Returns
        -------
        float
            The product of the branching fractions of all decays in the chain,
            with each sub-decay counted once per occurrence of its particle.

        Notes
        -----
        Calculation requires a flattening of the entire decay chain.
        """
        return self.flatten().bf

    @property
    def ndecays(self) -> int:
        """
        Return the number of decay modes including the top-level decay.

        Returns
        -------
        int
            The number of decay modes in the chain.
        """
        return len(self.decays)

    def to_string(self) -> str:
        """
        One-line string representation of the entire decay chain.

        Sub-decays are enclosed in round parentheses.

        Returns
        -------
        str
            The decay descriptor of the chain.

        Examples
        --------
        >>> dm1 = DecayMode(0.6770, "D0 pi+") # D*+
        >>> dm2 = DecayMode(0.0124, "K_S0 pi0") # D0
        >>> dm3 = DecayMode(0.692, "pi+ pi-") # K_S0
        >>> dm4 = DecayMode(0.98823, "gamma gamma") # pi0
        >>> dc = DecayChain("D*+", {"D*+":dm1, "D0":dm2, "K_S0":dm3, "pi0":dm4})
        >>> print(dc.to_string())
        D*+ -> (D0 -> (K_S0 -> pi+ pi-) (pi0 -> gamma gamma)) pi+
        """
        dc_dict = self.to_dict()
        descriptors = _expand_decay_modes(dc_dict, top=True)
        assert len(descriptors) == 1
        return descriptors[0]

    def print_as_tree(self) -> None:  # pragma: no cover
        """
        Tree-structure like print of the entire decay chain.

        Examples
        --------
        >>> dm1 = DecayMode(0.028, 'K_S0 pi+ pi-')
        >>> dm2 = DecayMode(0.692, 'pi+ pi-')
        >>> dc = DecayChain('D0', {'D0':dm1, 'K_S0':dm2})
        >>> dc.print_as_tree()
        D0
        +--> K_S0
        |    +--> pi+
        |    +--> pi-
        +--> pi+
        +--> pi-

        >>> dm1 = DecayMode(0.0124, 'K_S0 pi0')
        >>> dm2 = DecayMode(0.692, 'pi+ pi-')
        >>> dm3 = DecayMode(0.98823, 'gamma gamma')
        >>> dc = DecayChain('D0', {'D0':dm1, 'K_S0':dm2, 'pi0':dm3})
        >>> dc.print_as_tree()
        D0
        +--> K_S0
        |    +--> pi+
        |    +--> pi-
        +--> pi0
             +--> gamma
             +--> gamma

        >>> dm1 = DecayMode(0.6770, 'D0 pi+')
        >>> dm2 = DecayMode(0.0124, 'K_S0 pi0')
        >>> dm3 = DecayMode(0.692, 'pi+ pi-')
        >>> dm4 = DecayMode(0.98823, 'gamma gamma')
        >>> dc = DecayChain('D*+', {'D*+':dm1, 'D0':dm2, 'K_S0':dm3, 'pi0':dm4})
        >>> dc.print_as_tree()
        D*+
        +--> D0
        |    +--> K_S0
        |    |    +--> pi+
        |    |    +--> pi-
        |    +--> pi0
        |         +--> gamma
        |         +--> gamma
        +--> pi+
        """
        indent = 4
        arrow = "+--> "
        bar = "|"  # pylint: disable=disallowed-name

        # TODO: simplify logic and perform further checks
        def _print(
            decay_dict: DecayChainDict,
            depth: int = 0,
            link: bool = False,
            last: bool = False,
        ) -> None:
            mother = next(iter(decay_dict.keys()))
            prefix = bar if (link and depth > 1) else ""
            prefix = prefix + " " * indent * (depth - 1)
            for i_decay in decay_dict[mother]:
                print(prefix, arrow if depth > 0 else "", mother, sep="")  # noqa: T201
                fsps = i_decay["fs"]
                n = len(list(fsps))
                depth += 1
                for j, fsp in enumerate(fsps):
                    prefix = bar if (link and depth > 1) else ""
                    if last:
                        prefix = prefix + " " * indent * (depth - 1) + " "
                    else:
                        prefix = (prefix + " " * indent) * (depth - 1)
                    if isinstance(fsp, str):
                        print(prefix, arrow, fsp, sep="")  # noqa: T201
                    else:
                        _print(
                            fsp,
                            depth=depth,
                            link=(link or (j < n - 1)),
                            last=(j == n - 1),
                        )

        dc_dict = self.to_dict()
        _print(dc_dict)

    def to_dict(self) -> DecayChainDict:
        """
        Return the decay chain as a dictionary representation.

        The format is the same as ``DecFileParser.build_decay_chains(...)``.

        Returns
        -------
        dict
            The mother particle as single key, with its decay modes as value.
            Sub-decays are nested in the 'fs' lists.

        Examples
        --------
        >>> dm1 = DecayMode(0.028, 'K_S0 pi+ pi-')
        >>> dm2 = DecayMode(0.692, 'pi+ pi-')
        >>> dc = DecayChain('D0', {'D0':dm1, 'K_S0':dm2})
        >>> dc.to_dict()    # doctest: +NORMALIZE_WHITESPACE
        {'D0': [{'bf': 0.028,
            'fs': [{'K_S0': [{'bf': 0.692,
                'fs': ['pi+', 'pi-'],
                'model': '',
                'model_params': ''}]},
             'pi+',
             'pi-'],
            'model': '',
            'model_params': ''}]}
        """

        # Ideally this would be a recursive type, DecayDict = dict[str, list[str | DecayDict]]
        DecayDict = dict[str, list[Any]]

        def recursively_replace(mother: str) -> DecayDict:
            dm = self.decays[mother].to_dict()
            result = []
            list_fsp = dm["fs"]
            assert isinstance(list_fsp, list)

            for pos, fsp in enumerate(list_fsp):
                if fsp in self.decays:
                    list_fsp[pos] = recursively_replace(fsp)  # type: ignore[call-overload]
            result.append(dm)
            return {mother: result}

        return recursively_replace(self.mother)

    def flatten(
        self,
        stable_particles: Collection[str] = (),
    ) -> Self:
        """
        Flatten the decay chain, replacing intermediate particles with their final states.

        Parameters
        ----------
        stable_particles : collection of str, optional
            If provided, ignores the sub-decays of the listed particles,
            considering them as stable. The mother particle is always
            decayed (flattening its own decay is the whole point), so it
            is ignored even if present in this collection. Default is ().

        Returns
        -------
        DecayChain
            A new decay chain with a single decay mode, that of the mother
            particle, whose branching fraction is the visible branching fraction.

        Notes
        -----
        After flattening the only ``DecayMode`` metadata kept is that of the top-level decay,
        i.e. that of the mother particle (nothing else would make sense).

        Examples
        --------
        >>> dm1 = DecayMode(0.0124, 'K_S0 pi0', model='PHSP')
        >>> dm2 = DecayMode(0.692, 'pi+ pi-')
        >>> dm3 = DecayMode(0.98823, 'gamma gamma')
        >>> dc = DecayChain('D0', {'D0':dm1, 'K_S0':dm2, 'pi0':dm3})
        >>>
        >>> dc.flatten()
        <DecayChain: D0 -> gamma gamma pi+ pi- (0 sub-decays), BF=0.008479803984>
        >>> dc.flatten().to_dict()    # doctest: +NORMALIZE_WHITESPACE
        {'D0': [{'bf': 0.008479803984,
           'fs': ['gamma', 'gamma', 'pi+', 'pi-'],
           'model': 'PHSP',
           'model_params': ''}]}

        >>> dc.flatten(stable_particles=('K_S0', 'pi0')).decays
        {'D0': <DecayMode: daughters=K_S0 pi0, BF=0.0124>}
        """

        vis_bf = self.bf
        fs = DaughtersDict(self.decays[self.mother].daughters)

        if stable_particles:
            # The mother must always be decayed (that is the point of
            # flattening), so it is never treated as stable even if listed.
            keys = [
                k for k in self.decays if k == self.mother or k not in stable_particles
            ]
        else:
            keys = list(self.decays.keys())
        keys.insert(0, keys.pop(keys.index(self.mother)))

        further_to_replace = True
        while further_to_replace:
            for k in keys:
                if k in fs:
                    n_k = fs[k]
                    vis_bf *= self.decays[k].bf ** n_k
                    for _ in range(n_k):
                        fs += self.decays[k].daughters
                    fs[k] -= n_k
            further_to_replace = any(fs[_k] > 0 for _k in keys)

        return self.__class__(
            self.mother,
            {self.mother: DecayMode(vis_bf, fs, **self.top_level_decay().metadata)},
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.mother} -> {self.top_level_decay().daughters.to_string()} ({len(self.decays) - 1} sub-decays), BF={self.bf}>"

    def __str__(self) -> str:
        return repr(self)
