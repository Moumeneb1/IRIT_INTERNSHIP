import setuptools
from setuptools import find_packages


setuptools.setup(
    name='easy_nlp',
    version='0.4',
    author="Boumadane Abdelmoumene",
    author_email="fa_boumadane@esi.dz",
    description="An NLP Utility Package",
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Moumeneb1/IRIT_INTERNSHIP",
    # packages=['easy_nlp'],
    package_dir={"": "."},
    packages=find_packages("."),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "transformers",
        "torch==1.4.0",
        "treetaggerwrapper",
        "twitterscraper",
        "tensorboardX",
        "torchvision==0.5.0",
        "pandas-profiling",
        "torchsampler @ git+https://github.com/ufoym/imbalanced-dataset-sampler#egg=imbalanced-dataset-sampler",
    ],
)
