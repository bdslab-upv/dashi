from setuptools import setup, find_packages

setup(
    name="dashi",  # Replace with your library's name
    version="0.1.0",  # Initial version
    author="Carlos Sáez Silvestre, David Fernández Narro, Ángel Sánchez García, Pablo Ferri Borredá",
    author_email="",
    description="Dataset shifts analysis and characterization in python",
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bdslab-upv/dashi",
    packages=find_packages(),
    include_package_data=False,  # Include non-Python files listed in MANIFEST.in
    install_requires=[  # Add your dependencies here
        "numpy>=1.26.1",
        "pandas>=2.2.2",
        "plotly>=5.18.0",
        "sklearn==1.5.1",
        "dateutil>=2.8.2",
        "tqdm>=4.66.5",
        "prince>=0.14.0"
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
