from setuptools import setup, find_packages

setup(
    name="ml4h-latentverse",
    version="0.2.0",
    author="Yoanna Turura, Majd Alafrange",
    author_email="yturura@broadinstitute.org, majd@mit.edu",
    description="A library for evaluating self-supervised representations in ML",
    packages=find_packages(include=["src", "src"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)