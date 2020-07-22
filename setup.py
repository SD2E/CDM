from setuptools import setup, find_packages

DISTNAME = 'cdm'
DESCRIPTION = 'a tool for creating combinatorial design models and executing associated analyses'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'netrias'
MAINTAINER_EMAIL = 'eramian@netrias.com'

setup(
    name=DISTNAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    packages=['cdm'] + ['cdm/' + s for s in find_packages('cdm')],
    include_package_data=True,
    install_requires=[
        "keras",
        "pandas",
        "sklearn",
        "seaborn"
    ]
)
