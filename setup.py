from setuptools import setup, find_packages

setup(
    name="rual",  # Project name
    version="0.1.0",  # Initial version
    author="Laust Moesgaard",  # Your name
    author_email="moesgaard@sdu.dk",  # Your email
    description="Package for running Ramp Up Active Learning on an enumerated small molecule library.",
    long_description=open("README.md").read(),  # Ensure you have a README.md
    long_description_content_type="text/markdown",  # Specify README format
    url="https://github.com/lmoesgaard/rual",  # GitHub repository URL
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license if different
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Specify the minimum Python version
    install_requires=[
        "numpy",
        "pandas",
        "pyarrow",
        "rdkit-pypi",  # Ensure you're using the RDKit package compatible with pip
        "scikit-learn",  # sklearn is usually installed as scikit-learn
    ],
)
