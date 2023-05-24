import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="min_speckle",
    version="23.05.22",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="Classify speckle patterns with neural networks at scale.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/min-speckle",
    keywords = ['X-ray', 'Single particle imaging', 'Speckle patterns'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
