"""Classes and functions for computing multiple quality metrics."""

from __future__ import annotations


import warnings
from copy import deepcopy

import numpy as np

from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.sortinganalyzer import register_result_extension, AnalyzerExtension


from .quality_metric_list import compute_pc_metrics, _misc_metric_name_to_func, _possible_pc_metric_names
from .misc_metrics import _default_params as misc_metrics_params
from .pca_metrics import _default_params as pca_metrics_params


class ComputeQualityMetrics(AnalyzerExtension):
    """
    Compute quality metrics on a `sorting_analyzer`.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object.
    metric_names : list or None
        List of quality metrics to compute.
    qm_params : dict or None
        Dictionary with parameters for quality metrics calculation.
        Default parameters can be obtained with: `si.qualitymetrics.get_default_qm_params()`
    skip_pc_metrics : bool
        If True, PC metrics computation is skipped.

    Returns
    -------
    metrics: pandas.DataFrame
        Data frame with the computed metrics.

    Notes
    -----
    principal_components are loaded automatically if already computed.
    """

    extension_name = "quality_metrics"
    depend_on = ["templates", "noise_levels"]
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = True

    def _set_params(self, metric_names=None, qm_params=None, peak_sign=None, seed=None, skip_pc_metrics=False):
        if metric_names is None:
            metric_names = list(_misc_metric_name_to_func.keys())
            # if PC is available, PC metrics are automatically added to the list
            if self.sorting_analyzer.has_extension("principal_components") and not skip_pc_metrics:
                # by default 'nearest_neightbor' is removed because too slow
                pc_metrics = _possible_pc_metric_names.copy()
                pc_metrics.remove("nn_isolation")
                pc_metrics.remove("nn_noise_overlap")
                metric_names += pc_metrics
            # if spike_locations are not available, drift is removed from the list
            if not self.sorting_analyzer.has_extension("spike_locations"):
                if "drift" in metric_names:
                    metric_names.remove("drift")

        qm_params_ = get_default_qm_params()
        for k in qm_params_:
            if qm_params is not None and k in qm_params:
                qm_params_[k].update(qm_params[k])
            if "peak_sign" in qm_params_[k] and peak_sign is not None:
                qm_params_[k]["peak_sign"] = peak_sign

        params = dict(
            metric_names=[str(name) for name in np.unique(metric_names)],
            peak_sign=peak_sign,
            seed=seed,
            qm_params=qm_params_,
            skip_pc_metrics=skip_pc_metrics,
        )

        return params

    def _select_extension_data(self, unit_ids):
        new_metrics = self.data["metrics"].loc[np.array(unit_ids)]
        new_data = dict(metrics=new_metrics)
        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        import pandas as pd

        old_metrics = self.data["metrics"]

        all_unit_ids = new_sorting_analyzer.unit_ids
        not_new_ids = all_unit_ids[~np.isin(all_unit_ids, new_unit_ids)]

        metrics = pd.DataFrame(index=all_unit_ids, columns=old_metrics.columns)

        metrics.loc[not_new_ids, :] = old_metrics.loc[not_new_ids, :]
        metrics.loc[new_unit_ids, :] = self._compute_metrics(new_sorting_analyzer, new_unit_ids, verbose, **job_kwargs)

        new_data = dict(metrics=metrics)
        return new_data

    def _compute_metrics(self, sorting_analyzer, unit_ids=None, verbose=False, **job_kwargs):
        """
        Compute quality metrics.
        """
        metric_names = self.params["metric_names"]
        qm_params = self.params["qm_params"]
        # sparsity = self.params["sparsity"]
        seed = self.params["seed"]

        # update job_kwargs with global ones
        job_kwargs = fix_job_kwargs(job_kwargs)
        n_jobs = job_kwargs["n_jobs"]
        progress_bar = job_kwargs["progress_bar"]

        if unit_ids is None:
            sorting = sorting_analyzer.sorting
            unit_ids = sorting.unit_ids
            non_empty_unit_ids = sorting.get_non_empty_unit_ids()
            empty_unit_ids = unit_ids[~np.isin(unit_ids, non_empty_unit_ids)]
            if len(empty_unit_ids) > 0:
                warnings.warn(
                    f"Units {empty_unit_ids} are empty. Quality metrics will be set to NaN "
                    f"for these units.\n To remove empty units, use `sorting.remove_empty_units()`."
                )
        else:
            non_empty_unit_ids = unit_ids
            empty_unit_ids = []

        import pandas as pd

        metrics = pd.DataFrame(index=unit_ids)

        # simple metrics not based on PCs
        for metric_name in metric_names:
            # keep PC metrics for later
            if metric_name in _possible_pc_metric_names:
                continue
            if verbose:
                if metric_name not in _possible_pc_metric_names:
                    print(f"Computing {metric_name}")

            func = _misc_metric_name_to_func[metric_name]

            params = qm_params[metric_name] if metric_name in qm_params else {}
            res = func(sorting_analyzer, unit_ids=non_empty_unit_ids, **params)
            # QM with uninstall dependencies might return None
            if res is not None:
                if isinstance(res, dict):
                    # res is a dict convert to series
                    metrics.loc[non_empty_unit_ids, metric_name] = pd.Series(res)
                else:
                    # res is a namedtuple with several dict
                    # so several columns
                    for i, col in enumerate(res._fields):
                        metrics.loc[non_empty_unit_ids, col] = pd.Series(res[i])

        # metrics based on PCs
        pc_metric_names = [k for k in metric_names if k in _possible_pc_metric_names]
        if len(pc_metric_names) > 0 and not self.params["skip_pc_metrics"]:
            if not sorting_analyzer.has_extension("principal_components"):
                raise ValueError(
                    "To compute principal components base metrics, the principal components "
                    "extension must be computed first."
                )
            pc_metrics = compute_pc_metrics(
                sorting_analyzer,
                unit_ids=non_empty_unit_ids,
                metric_names=pc_metric_names,
                # sparsity=sparsity,
                progress_bar=progress_bar,
                n_jobs=n_jobs,
                qm_params=qm_params,
                seed=seed,
            )
            for col, values in pc_metrics.items():
                metrics.loc[non_empty_unit_ids, col] = pd.Series(values)

        # add NaN for empty units
        if len(empty_unit_ids) > 0:
            metrics.loc[empty_unit_ids] = np.nan

        return metrics

    def _run(self, verbose=False, **job_kwargs):
        self.data["metrics"] = self._compute_metrics(
            sorting_analyzer=self.sorting_analyzer, unit_ids=None, verbose=verbose, **job_kwargs
        )

    def _get_data(self):
        return self.data["metrics"]


register_result_extension(ComputeQualityMetrics)
compute_quality_metrics = ComputeQualityMetrics.function_factory()


def get_quality_metric_list():
    """
    Return a list of the available quality metrics.
    """

    return deepcopy(list(_misc_metric_name_to_func.keys()))


def get_default_qm_params():
    """
    Return default dictionary of quality metrics parameters.

    Returns
    -------
    dict
        Default qm parameters with metric name as key and parameter dictionary as values.
    """
    default_params = {}
    default_params.update(misc_metrics_params)
    default_params.update(pca_metrics_params)
    return deepcopy(default_params)
