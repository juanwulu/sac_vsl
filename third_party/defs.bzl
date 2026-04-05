"""Defines custom rules for Python binaries, libraries, and tests"""

load("@py_3_10_cpu//:requirements.bzl", cpu_req = "requirement")
load("@py_3_10_cuda//:requirements.bzl", cuda_req = "requirement")
load("@py_3_10_tpu//:requirements.bzl", tpu_req = "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

def _select_requirement(name):
    """Returns a requirement for the given name based on the platform.

    Args:
        name: The name of the requirement to return.

    Returns:
        A platform-specific requirement for the given name.
    """
    return select({
        "//third_party:is_cuda": [cuda_req(name)],
        "//third_party:is_tpu": [tpu_req(name)],
        "//conditions:default": [cpu_req(name)],
    })

def all_requirements(names = []):
    """Returns a list of all requirements for the given names.

    Args:
        names: A list of requirement names to include.

    Returns:
        A list of platform-specific requirements for the given names.
    """
    deps = []
    for name in names:
        deps += _select_requirement(name)

    if "fiddle" in names and "etils" not in names:
        deps += _select_requirement("etils")
        if "importlib_resources" not in names:
            deps += _select_requirement("importlib_resources")

    if "jax" in names:
        if "jaxlib" not in names:
            deps += _select_requirement("jaxlib")

    return deps

def ml_py_binary(name, **kwargs):
    """Creates a Python binary with all dependencies included."""
    native_deps, other_deps = _partition_deps(kwargs.pop("deps", []))
    kwargs["deps"] = native_deps + all_requirements(other_deps)

    return py_binary(
        name = name,
        **kwargs
    )

def ml_py_library(name, **kwargs):
    """Creates a Python library with all dependencies included."""
    native_deps, other_deps = _partition_deps(kwargs.pop("deps", []))
    kwargs["deps"] = native_deps + all_requirements(other_deps)

    return py_library(
        name = name,
        **kwargs
    )

def ml_py_test(name, **kwargs):
    """Creates a Python test with all dependencies included."""
    native_deps, other_deps = _partition_deps(kwargs.pop("deps", []))
    kwargs["deps"] = (
        native_deps +
        all_requirements(other_deps) +
        _select_requirement("pytest") +
        _select_requirement("pytest-cov")
    )

    return py_test(
        name = name,
        **kwargs
    )

def _partition_deps(deps = []):
    native_deps = []
    other_deps = []
    for dep in deps:
        if ":" in dep or dep.startswith("//"):
            native_deps.append(dep)
        else:
            other_deps.append(dep)

    return native_deps, other_deps
