from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = cythonize(
    [
        # Extension(
        #     "sglang.srt.mem_cache.tree_node",
        #     ["python/sglang/srt/mem_cache/tree_node.pyx"],
        #     language_level=3,
        # ),
        Extension(
            "sglang.srt.mem_cache.cy_evict",                   
            ["python/sglang/srt/mem_cache/cy_evict.pyx"],
            extra_compile_args=["-O3", "-march=native"]
        )
    ],
    language_level=3
)

setup(
    name="sglang",
    package_dir={"": "python"},          
    ext_modules=ext_modules,
    zip_safe=False,
)
