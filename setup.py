from setuptools import setup

setup(
    name="parode",
    version="0.0.1",
    author="Casian Iacob",
    author_email="casian.iacob@aalto.fi",
    install_requires=["jax[cuda12]", "matplotlib", "pandas"],
    zip_safe=False,
)
