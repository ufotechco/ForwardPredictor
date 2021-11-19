import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forwardpredictor",
    version="1.0.0",
    author="UFOTECH",
    author_email="gerencia@ufotech.co",
    description="A forward predictor regarding the data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ufotechco/ForwardPredictor",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)