from __future__ import annotations

import pytest

from spikeinterface.core import generate_ground_truth_recording, create_sorting_analyzer

job_kwargs = dict(n_jobs=-1)


def make_sorting_analyzer(sparse=True, num_units=5):
    recording, sorting = generate_ground_truth_recording(
        durations=[300.0],
        sampling_frequency=30000.0,
        num_channels=4,
        num_units=num_units,
        generate_sorting_kwargs=dict(firing_rates=20.0, refractory_period_ms=4.0),
        noise_kwargs=dict(noise_levels=5.0, strategy="on_the_fly"),
        seed=2205,
    )

    channel_ids_as_integers = [id for id in range(recording.get_num_channels())]
    unit_ids_as_integers = [id for id in range(sorting.get_num_units())]
    recording = recording.rename_channels(new_channel_ids=channel_ids_as_integers)
    sorting = sorting.rename_units(new_unit_ids=unit_ids_as_integers)

    sorting_analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format="memory", sparse=sparse)
    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates")
    sorting_analyzer.compute("noise_levels")
    # sorting_analyzer.compute("principal_components")
    # sorting_analyzer.compute("template_similarity")
    # sorting_analyzer.compute("quality_metrics", metric_names=["snr"])

    return sorting_analyzer


@pytest.fixture(scope="module")
def sorting_analyzer_for_curation():
    return make_sorting_analyzer(sparse=True)


if __name__ == "__main__":
    sorting_analyzer = make_sorting_analyzer(sparse=False)
    print(sorting_analyzer)
