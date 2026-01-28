from setuptools import setup, find_packages

setup(
    name="rual",  
    version="0.1.0", 
    author="Laust Moesgaard",  
    author_email="moesgaard@sdu.dk", 
    description="Package for running Ramp Up Active Learning on an enumerated small molecule library.",
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown", 
    url="https://github.com/lmoesgaard/rual", 
    packages=find_packages(),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8", 
    install_requires=[
        "numpy",
        "pandas",
        "pyarrow",
        "rdkit", 
        "scikit-learn",  
    ],
    entry_points={
        "console_scripts": [
            "rual-dbbuilder=rual.database.dbbuilder:main",
            "rual-smina=rual.scoring.smina_cli:main",
        ]
    },
)
