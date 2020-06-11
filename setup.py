import setuptools


setuptools.setup(
     name='Easy NLP',  
     version='0.1',
     scripts=[''] ,
     author="Boumadane Abdelmoumene",
     author_email="fa_boumadane@esi.dz",
     description="An NLP Utility Package",
     long_description=open('README.txt').read(),
     long_description_content_type="text/markdown",
     url="https://github.com/Moumeneb1/IRIT_INTERNSHIP",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     install_requires=[
       "Transformers >= 1.1.1",
       "Pandas",
       "Numpy",
   ],
 )