.. _regularized-kde:

Regularized multivariate KDE
========================================

When to use it
--------------------

Multivariate maps fit a Gaussian kernel density estimate (KDE) separately to each
site or temporal batch after fitting a shared dimensionality reduction. Enough
rows do not guarantee an invertible covariance: a site can contain identical
points or points on a line or plane. Pooled PCA does not guarantee full rank
within every group. Low within-group variability does not necessarily imply low
divergence between groups.

The following functions accept ``regularization_strength``:

* ``estimate_multivariate_data_source_map``
* ``estimate_conditional_data_source_map``
* ``estimate_multivariate_data_temporal_map``
* ``estimate_conditional_data_temporal_map``

This option concerns multivariate KDE maps, not the separate univariate APIs.

Default behavior
----------------

``regularization_strength=0.0`` (the default) preserves the unregularized KDE.
Singular eligible source groups raise an informative error. Singular eligible
temporal batches warn and return NaN maps. Both messages suggest enabling
regularization or reducing dimensions where supported. Groups with at most as
many rows as retained dimensions keep their existing missing-map behavior.

For a 3D analysis, trying ``dimensions=2`` can help, but is not guaranteed to fix
within-group degeneracy and changes the comparison space. The nonconditional
multivariate APIs do not accept ``dimensions=1``. Changing ``kde_resolution`` alone
does not repair a singular covariance.

Definition and units
--------------------

For group :math:`g`, positive regularization uses Gaussian kernels centered on the
original reduced observations with covariance

.. math::

   H_g = h_g^2 S_g + \tau^2 T, \qquad
   T = \operatorname{diag}(v_1,\ldots,v_d), \qquad
   h_g = \left(\frac{n_g(d+2)}{4}\right)^{-1/(d+4)}.

Here :math:`S_g` is the sample covariance (denominator :math:`n_g-1`),
:math:`h_g` is Silverman's bandwidth factor, and :math:`v_j` is the pooled sample
variance of reduced coordinate :math:`j`. The parameter
``regularization_strength`` is :math:`\tau`, a finite non-negative,
dimensionless **standard-deviation multiplier**. The actual added covariance is
:math:`\tau^2 T`, applied **after** bandwidth scaling.

For example, ``regularization_strength=0.05`` adds variance
:math:`0.0025v_j`, equivalent to Gaussian smoothing standard deviation
:math:`0.05\sqrt{v_j}` along coordinate :math:`j`. It does **not** add 0.05 to the
diagonal and does not imply a 5% change in divergence. This is scale-aware ridge,
equivalent to ordinary ridge in coordinates standardized by pooled standard
deviations. No random noise is added to observations.

Every estimated group receives the same additive covariance, including groups
whose unregularized KDE succeeds. For conditional maps, the reference is computed
before splitting by label; all labels share the additive covariance and grid.
Larger groups contribute more observations to the pooled reference. Temporal
analysis uses the pooled analysis data, not a past-only or online reference.
The same strength in separate calls need not give the same absolute smoothing:
their fitted projections and pooled variances may differ.

Nonempty sparse, collinear and constant groups are retained when regularization
is positive. For a singleton, the empirical covariance contribution is explicitly
defined as zero. Empty groups remain missing. Warnings identify sparse or
rank-deficient groups: numerical estimability does not remove sampling uncertainty.
If a retained coordinate has effectively zero pooled variance, the function
raises an error asking you to review variables or reduce dimensions where
supported; it does not silently invent a variance or drop a coordinate.

Interpretation and choosing a strength
--------------------------------------------------

Regularization changes densities and consequently divergence estimates. Interpret
the result as **differences between distributions at the chosen smoothing scale**.
Larger strengths suppress more fine-scale distinctions. Choose the scale from
measurement precision or scientifically meaningful differences if possible.
Otherwise, 0.05 is an exploratory starting point, **not a validated universal
default**. Compare strengths such as 0.02, 0.05 and 0.10 and report whether
important groupings, outliers and relative divergences are stable. Record the
strength, dimensions, reduction method, scaling and grid resolution with results.

Usage
-----

The examples assume prepared DataFrames: ``joint_data`` has covariates and
``site``; ``temporal_data`` has covariates and datetime column ``date``; the two
conditional frames additionally have ``label``. Do not accidentally pass site
identifiers, dates or labels as covariates in another analysis.

.. code-block:: python

   import dashi as ds

   settings = dict(dimensions=2, dim_reduction="PCA", scale=True,
                   regularization_strength=0.05, kde_resolution=50)

   source_map = ds.estimate_multivariate_data_source_map(
       joint_data, source_column_name="site", **settings)
   temporal_map = ds.estimate_multivariate_data_temporal_map(
       temporal_data, date_column_name="date", period="month", **settings)
   source_by_label = ds.estimate_conditional_data_source_map(
       conditional_source_data, source_column_name="site",
       label_column_name="label", **settings)
   temporal_by_label = ds.estimate_conditional_data_temporal_map(
       conditional_temporal_data, date_column_name="date",
       label_column_name="label", period="month", **settings)

Grid accuracy
-------------

Positive regularization extends shared bounds to at least four kernel marginal
standard deviations beyond each group's extreme observations. Densities are
evaluated in log space and normalized over the finite grid to produce the map's
discrete probabilities. The finite support still truncates Gaussian tails.

A warning is issued if grid spacing exceeds any group's conditional kernel
standard deviation. This is a diagnostic, not an accuracy guarantee. Even 50
points per coordinate may be insufficient; the default 10 can be particularly
coarse for narrow kernels. Compare, for example, resolutions 50 and 100 at a fixed
strength, then refine until your divergence conclusions stabilize. Cost grows
with ``kde_resolution ** dimensions`` and the observation count. The library does
not automatically increase resolution. Do not increase smoothing merely to make
a coarse grid work. Grids at different resolutions cannot be compared cell by
cell; compare the resulting summaries instead.
