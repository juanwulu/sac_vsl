load("//third_party:defs.bzl", "ml_py_binary", "ml_py_library")

package(default_visibility = ["//sac_vsl:__subpackages__"])

ml_py_binary(
    name = "train_sac",
    srcs = ["scripts/train_sac.py"],
    deps = [
        "ray",
        "torch",
        ":env",
    ],
)

ml_py_library(
    name = "env",
    srcs = ["env.py"],
    deps = [
        "gym",
        "numpy",
    ],
)

ml_py_library(
    name = "error",
    srcs = ["error.py"],
)

ml_py_library(
    name = "memory",
    srcs = ["memory.py"],
    deps = ["numpy"],
)

ml_py_library(
    name = "meter",
    srcs = ["meter.py"],
)

ml_py_library(
    name = "typing",
    srcs = ["typing.py"],
)
