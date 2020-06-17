import setuptools


setuptools.setup(
    name='EasyNLP',
    version='0.2',
    author="Boumadane Abdelmoumene",
    author_email="fa_boumadane@esi.dz",
    description="An NLP Utility Package",
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Moumeneb1/IRIT_INTERNSHIP",
    packages=['EasyNLP'],
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
        "torch",
        "treetaggerwrapper",
        "twitterscraper",
        "tensorboardX",
    ],
)
