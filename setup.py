from setuptools import setup, find_packages

setup(
    name="dashi",  # Replace with your library's name
    version="0.1.0",  # Initial version
    author="Carlos Sáez Silvestre, David Fernández Narro, Ángel Sánchez García",
    author_email="",
    description="A brief description of your library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bdslab-upv/dashi",
    packages=find_packages(),
    include_package_data=False,  # Include non-Python files listed in MANIFEST.in
    install_requires=[  # Add your dependencies here
        # Example: "numpy>=1.21.0",
        "numpy>=1.26.2",
        "pandas>=2.2.3",
        "plotly==5.18.0",
        "sklearn==",
        "dateutil",
        "tqdm",
        "prince",

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust if not MIT
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",  # Minimum Python version required
    keywords="dashi, datashift",  # Keywords for PyPI search
    project_urls={  # Additional URLs
        "Documentation": "https://github.com/bdslab-upv/dashi/docs",
        "Source": "https://github.com/bdslab-upv/dashi",
        "Tracker": "https://github.com/bdslab-upv/dashi/issues",
    },
)
