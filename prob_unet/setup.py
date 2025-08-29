from setuptools import setup, find_packages

setup(
    name='prob_unet',
    version='0.0.1',
    description='Utilities for probabilistic UNet project',
    author='Blaise Gauvin St-Denis',
    author_email='bstdenis@gmail.com',
    packages=find_packages(),
    keywords='machine learning, climate',
    python_requires='>=3.10',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development",
    ],
    package_data={'prob_unet': []}
)
