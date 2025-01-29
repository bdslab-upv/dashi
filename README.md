# dashi

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg) 
![Python Version](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)

Dataset shift analysis and characterization in python
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

- **Data Temporal Heatmaps (DTHs):** Provide an exploratory visualization for temporal shifts in data distributions.
- **Information Geometric Temporal (IGT) plots:** Offer a more sophisticated view of temporal data variability by means of embedding temporal batches in their latent statistical manifolds.
- **Multi-batch contingency matrices:** Compare multiple evaluation metrics (F1-Score, Recall, Precision, AUC, etc.) across training-test combinations between pairwise batches, either temporal or multi-source.

## Installation

You can install `dashi` using pip:

```bash
pip install dashi
```

Or install from source:

```bash
git clone https://github.com/bdslab-upv/dashi
cd dashi
pip install .
```

## Usage & Examples

Here’s a basic example of how to use `dashi`:

Find more examples in the [examples](examples/) directory.


## Documentation

Detailed documentation is available at [WIP](https://example.com/docs).

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.


```
Copyright 2024 Biomedical Data Science Lab, ITACA Institute, Universitat Politècnica de València (Spain)

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
Part of the Python library `dashi` has been inspired by the R [EHRtemporalVariability](https://CRAN.R-project.org/package=EHRtemporalVariability) package, licensed under the Apache 2.0 License, and authored by part of this `dashi` library authors.

## Authorship

- **Authors:** David Fernández Narro (UPV), Pablo Ferri Borredá (UPV), Ángel Sánchez-García (UPV), Juan M García-Gómez (UPV), Carlos Sáez (UPV)

- **Contact:** carsaesi@upv.es

## Acknowledgements

Funded by Agencia Estatal de Investigación—Proyectos de Generación de Conocimiento 2022, project KINEMAI (PID2022-138636OA-I00). 

## References
1. Sáez, C., Rodrigues, P. P., Gama, J., Robles, M., & García-Gómez, J. M. (2015). Probabilistic change detection and visualization methods for the assessment of temporal stability in biomedical data quality. Data Mining and Knowledge Discovery, 29(4), 950-975. https://doi.org/10.1007/s10618-014-0378-6
2. Sáez, C., & García-Gómez, J. M. (2018). Kinematics of Big Biomedical Data to characterize temporal variability and seasonality of data repositories: Functional Data Analysis of data temporal evolution over non-parametric statistical manifolds. International Journal of Medical Informatics, 119, 109-124. https://doi.org/10.1016/j.ijmedinf.2018.09.015
3. Sáez, C., Zurriaga, O., Pérez-Panadés, J., Melchor, I., Robles, M., & García-Gómez, J. M. (2016). Applying probabilistic temporal and multisite data quality control methods to a public health mortality registry in Spain: A systematic approach to quality control of repositories. Journal of the American Medical Informatics Association, 23(6), 1085-1095. https://doi.org/10.1093/jamia/ocw010
4. Sáez C, Gutiérrez-Sacristán A, Kohane I, García-Gómez JM, Avillach P. EHRtemporalVariability: delineating temporal data-set shifts in electronic health records. GigaScience, Volume 9, Issue 8, August 2020, giaa079. https://doi.org/10.1093/gigascience/giaa079
5. Sáez, C., Robles, M. and García-Gómez, J.M., 2017. Stability metrics for multi-source biomedical data based on simplicial projections from probability distribution distances. Statistical methods in medical research. 2017;26(1):312-336. https://doi.org/10.1177/0962280214545122


