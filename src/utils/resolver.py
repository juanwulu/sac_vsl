# =============================================================================
# @file   resolver.py
# @author Juanwu Lu
# @date   Nov-11-22
# =============================================================================
"""Resolver helper functions"""
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/resolver.py
from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Union


def _normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(classes: Sequence[Any],
             class_dict: Dict[str, Any],
             query: Union[str, Any],
             base_cls: Optional[Any] = None,
             base_cls_repr: Optional[str] = None,
             *args, **kwargs) -> Callable:
    if not isinstance(query, str):
        warnings.warn(f'Expect query to be `str`, but got {type(query):s}.')
        return query

    query_repr = _normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls is not None else ''
    base_cls_repr = _normalize_string(base_cls_repr)

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    for cls in classes:
        cls_repr = _normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f'Failed to resolve {query:s} among choices {choices}.')
