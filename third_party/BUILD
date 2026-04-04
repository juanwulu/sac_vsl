load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_uv//uv:pip.bzl", "pip_compile")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "is_cpu",
    define_values = {"ml_platform": "cpu"},
)

config_setting(
    name = "is_cuda",
    define_values = {"ml_platform": "cuda"},
)

config_setting(
    name = "is_tpu",
    define_values = {"ml_platform": "tpu"},
)

config_setting(
    name = "is_mps",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
    ],
    define_values = {"ml_platform": "mps"},
)

pip_compile(
    name = "requirements_cpu",
    args = [
        "--generate-hashes",
        "--emit-find-links",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_3_10_cpu.txt",
)

compile_pip_requirements(
    name = "requirements_cuda",
    timeout = "moderate",
    extra_args = [
        "--allow-unsafe",
        "--resolver=backtracking",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_3_10_cuda.txt",
)

pip_compile(
    name = "requirements_tpu",
    args = [
        "--generate-hashes",
        "--emit-find-links",
    ],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_3_10_tpu.txt",
)

pip_compile(
    name = "requirements_mps",
    args = [
        "--generate-hashes",
        "--emit-find-links",
    ],
    python_platform = "aarch64-apple-darwin",
    requirements_in = "requirements.in",
    requirements_txt = "requirements_3_10_mps.txt",
)
