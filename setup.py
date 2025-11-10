from setuptools import setup, find_packages

setup(
    name="orkeslora",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyyaml",
        "paramiko",
        "scikit-learn",  # optional, for example scripts
        "mlflow"
    ],
    entry_points={
        "console_scripts": [
            "orkeslora=orkeslora.cli:main"
        ]
    },
    python_requires=">=3.8",
)
