from setuptools import setup, find_packages

setup(
    name='clustering_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib'
    ],
    description='A Python package for KMeans and DBSCAN clustering algorithms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Manus',
    author_email='manus@example.com',
    url='https://github.com/manus/clustering_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    python_requires='>=3.6',
)


