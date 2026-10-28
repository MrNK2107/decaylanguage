# Copyright (c) 2018-2026, Eduardo Rodrigues and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/decaylanguage for details.

"""Check that the docstrings of decaylanguage.decay.decay follow the numpydoc format."""

from __future__ import annotations

import inspect

import pytest

from decaylanguage.decay import decay

validate = pytest.importorskip("numpydoc.validate").validate

MODULE = decay.__name__

# Section and parameter checks from
# https://numpydoc.readthedocs.io/en/latest/validation.html#built-in-validation-checks
CHECKS = {
    "GL06",  # unknown section, e.g. "Note" instead of "Notes"
    "GL07",  # sections in the wrong order
    "PR01",  # parameter not documented
    "PR02",  # unknown parameter documented
    "PR10",  # "name: type" instead of "name : type"
    "RT01",  # no Returns section
    "SS01",  # no summary
    "SS03",  # summary does not end with a period
    "SS06",  # summary does not fit in a single line
}


def _documented_names() -> list[str]:
    names = []
    for name, obj in vars(decay).items():
        if getattr(obj, "__module__", None) != MODULE or not callable(obj):
            continue
        if obj.__doc__:
            names.append(f"{MODULE}.{name}")
        if inspect.isclass(obj):
            for attr, member in vars(obj).items():
                func = member.__func__ if isinstance(member, classmethod) else member
                if isinstance(func, property):
                    func = func.fget
                if callable(func) and func.__doc__:
                    names.append(f"{MODULE}.{name}.{attr}")
    return names


@pytest.mark.parametrize("name", _documented_names())
def test_numpydoc_format(name: str) -> None:
    errors = [
        f"{code}: {msg}" for code, msg in validate(name)["errors"] if code in CHECKS
    ]
    assert not errors, "\n".join(errors)
