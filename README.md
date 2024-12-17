# dashi

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg) 
![Python Version](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)

Dataset shifts analysis and characterization in python
## What is `dashi`?
`dashi` is a Python library designed to **analyze and characterize temporal and multi-source dataset shifts**. It provides 
robust tools for both **supervised and unsupervised evaluation of dataset shifts,** empowering users to detect, understand, 
and address changes in data distributions with confidence.

### Key Features:

- **Supervised Characterization:**
Enables users to create classification or regression models using Random Forests trained on batched data 
(temporal or multi-source). This allows for the detailed analysis of how dataset shifts impact model performance, 
helping to pinpoint areas of potential degradation.
- **Unsupervised Characterization:** 
Facilitates the identification of temporal dataset shifts by projecting and visualizing data dissimilarities across time. 
This process involves:
  - Estimating data statistical distributions over time.
  - Projecting these distributions onto non-parametric statistical manifolds. These projections reveal patterns of
  latent temporal variability in the data, uncovering hidden trends and shifts.

### Visualization Tools:
To aid exploration and interpretation of dataset shifts, `dashi` includes visual analytics features such as:

- **Data Temporal Heatmaps (DTHs):** Provide an intuitive visualization of temporal data changes.
- **Information Geometric Temporal (IGT) plots:** Offer a more sophisticated view of data variability through statistical
manifolds projections.

## References
1. Roschewitz, M., Mehta, R., Jones, C., & Glocker, B. (2024). Automatic dataset shift identification to support root cause analysis of AI performance drift (arXiv:2411.07940). arXiv. https://doi.org/10.48550/arXiv.2411.07940
2. Sáez, C., & García-Gómez, J. M. (2018). Kinematics of Big Biomedical Data to characterize temporal variability and seasonality of data repositories: Functional Data Analysis of data temporal evolution over non-parametric statistical manifolds. International Journal of Medical Informatics, 119, 109-124. https://doi.org/10.1016/j.ijmedinf.2018.09.015
3. Sáez, C., Rodrigues, P. P., Gama, J., Robles, M., & García-Gómez, J. M. (2015). Probabilistic change detection and visualization methods for the assessment of temporal stability in biomedical data quality. Data Mining and Knowledge Discovery, 29(4), 950-975. https://doi.org/10.1007/s10618-014-0378-6
4. Sáez, C., Zurriaga, O., Pérez-Panadés, J., Melchor, I., Robles, M., & García-Gómez, J. M. (2016). Applying probabilistic temporal and multisite data quality control methods to a public health mortality registry in Spain: A systematic approach to quality control of repositories. Journal of the American Medical Informatics Association, 23(6), 1085-1095. https://doi.org/10.1093/jamia/ocw010


## Installation

You can install `dashi` using pip:

```bash
pip install dashi
```

Or install from source:

```bash
git clone https://github.com/yourusername/my_library.git
cd my_library
pip install .
```

## Usage

Here’s a basic example of how to use `my_library`:

```python
from my_library import some_function

# Example usage
result = some_function(args)
print(result)
```

### More Examples

Find more examples in the [examples](examples/) directory.

## Documentation

Detailed documentation is available at [Documentation Link](https://example.com/docs).

## Contributing

Contributions are welcome! Please check out the [contributing guidelines](CONTRIBUTING.md) for more details.

### Running Tests

To run tests, first install the testing dependencies:

```bash
pip install -r requirements.txt
```

Then run the tests using `pytest`:

```bash
pytest
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.


```
Copyright 2024 Biomedical Data Science Lab, Universitat Politècnica de València (Spain)

Licensed to the Apache Software Foundation (ASF) under one or more contributor
license agreements. See the NOTICE file distributed with this work for
additional information regarding copyright ownership. The ASF licenses this
file to you under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.
```

## Acknowledgments

This library, `dashi`, has been inspired by the [EHRtemporalVariability](https://github.com/hms-dbmi/EHRtemporalVariability) project, originally implemented in R by the **Harvard Medical School DBMI**. We adapted and extended its core concepts for Python to facilitate dataset shift analysis and characterization.

---

Made with ❤️ by [BDSLab](https://bdslab.upv.es)
