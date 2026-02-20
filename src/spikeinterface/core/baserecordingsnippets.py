from __future__ import annotations
from pathlib import Path

import numpy as np

from probeinterface import Probe, ProbeGroup, write_probeinterface, read_probeinterface, select_axes

from .base import BaseExtractor
from .recording_tools import check_probe_do_not_overlap

from warnings import warn


class BaseRecordingSnippets(BaseExtractor):
    """
    Mixin that handles all probe and channel operations
    """

    def __init__(self, sampling_frequency: float, channel_ids: list[str, int], dtype: np.dtype):
        BaseExtractor.__init__(self, channel_ids)
        self._sampling_frequency = float(sampling_frequency)
        self._dtype = np.dtype(dtype)
        self._probe_group = None
        self._channel_to_contact_indices = None
        self._channel_to_contact_ids = None

    @property
    def channel_ids(self):
        return self._main_ids

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def dtype(self):
        return self._dtype

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_channel_ids(self):
        return self._main_ids

    def get_num_channels(self):
        return len(self.get_channel_ids())

    def get_dtype(self):
        return self._dtype

    def has_scaleable_traces(self) -> bool:
        if self.get_property("gain_to_uV") is None or self.get_property("offset_to_uV") is None:
            return False
        else:
            return True

    # --- Property overrides for computed contact_vector ---
    # NOTE: This entire block exists only to keep `recording.get_property("contact_vector")`
    # working for backward compatibility. In my opinion no users rely on this directly;
    # the proper APIs are get_probe(), get_probegroup(), get_channel_locations(), and
    # get_property("location"). If we drop this compatibility layer, all the overrides
    # below (_compute_contact_vector, _compute_contact_vector_from, and multiple fallback
    # branches elsewhere) can be removed, simplifying the implementation considerably.

    def get_property(self, key, ids=None):
        if key == "contact_vector" and self._probe_group is not None:
            arr = self._compute_contact_vector()
            if ids is not None:
                inds = self.ids_to_indices(ids)
                arr = arr[inds]
            return arr
        return super().get_property(key, ids=ids)

    def get_property_keys(self):
        keys = super().get_property_keys()
        if self._probe_group is not None and "contact_vector" not in keys:
            keys = keys + ["contact_vector"]
        return keys

    def set_property(self, key, values, ids=None, missing_value=None):
        if key == "contact_vector" and self._probe_group is not None:
            return  # no-op: contact_vector is computed from _probe_group + mapping
        super().set_property(key, values, ids=ids, missing_value=missing_value)

    def _compute_contact_vector(self):
        """Reconstruct the contact_vector structured array from _probe_group + mapping."""
        probe_as_numpy = self._probe_group.to_numpy(complete=True)
        probe_as_numpy = probe_as_numpy[self._channel_to_contact_indices]
        probe_as_numpy = probe_as_numpy.copy()
        probe_as_numpy["device_channel_indices"] = np.arange(len(probe_as_numpy), dtype="int64")
        return probe_as_numpy

    # --- Probe presence checks ---

    def has_probe(self) -> bool:
        return self._probe_group is not None or "contact_vector" in super().get_property_keys()

    def has_channel_location(self) -> bool:
        return self.has_probe() or "location" in self.get_property_keys()

    def is_filtered(self):
        # the is_filtered is handle with annotation
        return self._annotations.get("is_filtered", False)

    # --- Probe setting ---

    def set_probe(self, probe, group_mode="auto", in_place=False, channel_to_contact_ids=None):
        """
        Attach a list of Probe object to a recording.

        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the recording
        group_mode: "auto" | "by_probe" | "by_shank" | "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks and two sides are present.
        in_place: bool
            False by default.
            Useful internally when extractor do self.set_probegroup(probe)
        channel_to_contact_ids : dict or None, default: None
            Optional mapping from channel_id to contact_id. If None, identity
            mapping is assumed (channel i reads from contact i).

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSlice or clone or itself)
        """
        assert isinstance(probe, Probe), "must give Probe"
        probegroup = ProbeGroup()
        probegroup.add_probe(probe)
        return self._set_probes(
            probegroup,
            group_mode=group_mode,
            in_place=in_place,
            channel_to_contact_ids=channel_to_contact_ids,
        )

    def set_probegroup(
        self,
        probegroup,
        group_mode="auto",
        in_place=False,
        channel_to_contact_ids=None,
    ):
        return self._set_probes(
            probegroup,
            group_mode=group_mode,
            in_place=in_place,
            channel_to_contact_ids=channel_to_contact_ids,
        )

    def get_channel_to_contact_ids(self):
        """Get the mapping from channel_ids to contact_ids.

        Returns
        -------
        channel_to_contact_ids : dict
            Mapping from channel_id to contact_id, or None if no probe is set.
        """
        return self._channel_to_contact_ids

    def _build_channel_to_contact_indices_from_ids(self, probegroup, channel_to_contact_ids):
        """Convert the user-provided id-based mapping to the internal index-based mapping.

        The user provides a dict {channel_id: contact_id} which is human-readable
        and uses stable identifiers. Internally, the recording stores an integer
        array (channel_to_contact_indices) where each entry is a flat index into
        the concatenated contacts of the ProbeGroup. This method bridges the two
        representations.

        Parameters
        ----------
        probegroup : ProbeGroup
            The probe group containing all probes.
        channel_to_contact_ids : dict
            User-provided mapping from channel_id to contact_id.

        Returns
        -------
        channel_to_contact_indices : np.ndarray
            Internal integer array of length num_channels, where entry i is the
            flat contact index in the ProbeGroup that channel i reads from.
        """
        # Build global contact_id -> flat index across all probes
        contact_id_to_index = {}
        offset = 0
        for probe in probegroup.probes:
            if probe.contact_ids is None:
                raise ValueError(
                    "Cannot use channel_to_contact_ids with probes that have no contact_ids set. "
                    "Call probe.set_contact_ids() first."
                )
            for index, contact_id in enumerate(probe.contact_ids):
                contact_id_str = str(contact_id)
                if contact_id_str in contact_id_to_index:
                    raise ValueError(
                        f"contact_id '{contact_id_str}' appears in multiple probes. "
                        f"contact_ids must be unique across the entire ProbeGroup."
                    )
                contact_id_to_index[contact_id_str] = offset + index
            offset += len(probe.contact_ids)

        # Build channel_id lookup using string keys for type-safe comparison
        # (JSON deserialization may produce string keys even for integer channel_ids)
        # Another reason to not use integers for ids : )
        channel_id_str_to_index = {str(cid): i for i, cid in enumerate(self.channel_ids)}
        channel_to_contact_indices = np.full(self.get_num_channels(), -1, dtype="int64")
        for channel_id, contact_id in channel_to_contact_ids.items():
            channel_id_str = str(channel_id)
            contact_id_str = str(contact_id)
            if channel_id_str not in channel_id_str_to_index:
                raise ValueError(f"channel_id '{channel_id_str}' not found in recording channel_ids.")
            if contact_id_str not in contact_id_to_index:
                raise ValueError(f"contact_id '{contact_id_str}' not found in ProbeGroup contact_ids.")
            channel_to_contact_indices[channel_id_str_to_index[channel_id_str]] = contact_id_to_index[contact_id_str]

        # Validate: every channel must be mapped (no -1 remaining)
        if np.any(channel_to_contact_indices < 0):
            unmapped = self.channel_ids[channel_to_contact_indices < 0]
            raise ValueError(
                f"Channels {unmapped} have no mapped contact. " f"Use select_channels() to exclude them first."
            )
        return channel_to_contact_indices

    def _build_channel_to_contact_indices_from_device_indices(self, probegroup):
        """Convert device_channel_indices from probes to internal mapping.

        Handles the legacy path where device_channel_indices is set on probes.
        Filters out -1 (unconnected), sorts by device indices.

        Parameters
        ----------
        probegroup : ProbeGroup
            The probe group with device_channel_indices set.

        Returns
        -------
        channel_to_contact_indices : np.ndarray
            Integer mapping array.
        new_channel_ids : np.ndarray
            The channel_ids that correspond to the mapping (may be a subset
            if some contacts were unconnected).
        """
        probe_as_numpy_array = probegroup.to_numpy(complete=True)

        # keep only connected contacts (!= -1)
        keep = probe_as_numpy_array["device_channel_indices"] >= 0
        if np.any(~keep):
            warn("The given probes have unconnected contacts: they are removed")

        probe_as_numpy_array = probe_as_numpy_array[keep]

        device_channel_indices = probe_as_numpy_array["device_channel_indices"]
        order = np.argsort(device_channel_indices)
        device_channel_indices = device_channel_indices[order]

        # validate range
        number_of_device_channel_indices = np.max(list(device_channel_indices) + [0])
        if number_of_device_channel_indices >= self.get_num_channels():
            error_msg = (
                f"The given Probe either has 'device_channel_indices' that does not match channel count \n"
                f"{len(device_channel_indices)} vs {self.get_num_channels()} \n"
                f"or it's max index {number_of_device_channel_indices} is the same as the number of channels "
                f"{self.get_num_channels()} \n"
                f"If using all channels remember that python is 0-indexed so max device_channel_index should be "
                f"{self.get_num_channels() - 1} \n"
                f"device_channel_indices are the following: {device_channel_indices} \n"
                f"recording channels are the following: {self.get_channel_ids()} \n"
            )
            raise ValueError(error_msg)

        new_channel_ids = self.get_channel_ids()[device_channel_indices]

        # Build the channel-to-contact mapping:
        # After filtering and sorting, probe_as_numpy_array[order] gives contacts in channel order.
        # We need to map from channel position -> original flat contact index in the full probegroup.
        #
        # The `keep` mask gives us which contacts from the full probegroup are connected.
        # `order` gives us the sorting of those connected contacts by device_channel_indices.
        # So the connected contact indices in original probegroup order are: np.where(keep)[0]
        # And after sorting: np.where(keep)[0][order]
        connected_contact_indices = np.where(keep)[0]
        channel_to_contact_indices = connected_contact_indices[order]

        return channel_to_contact_indices, new_channel_ids

    @staticmethod
    def _ensure_probes_have_contact_ids(probegroup):
        """Ensure every probe in a ProbeGroup has contact_ids set.

        If a probe has no contact_ids, assigns sequential string ids
        starting from the current global offset across the ProbeGroup.
        """
        offset = 0
        for probe in probegroup.probes:
            n = probe.get_contact_count()
            if probe.contact_ids is None:
                probe.set_contact_ids([str(offset + i) for i in range(n)])
            offset += n

    def _set_probes(
        self,
        probe_or_probe_group,
        group_mode="auto",
        in_place=False,
        channel_to_contact_ids=None,
    ):
        """
        Attach a list of Probe objects to a recording.
        For this Probe.device_channel_indices is used to link contacts to recording channels.
        If some contacts of the Probe are not connected (device_channel_indices=-1)
        then the recording is "sliced" and only connected channel are kept.

        The probe order is not kept. Channel ids are re-ordered to match the channel_ids of the recording.


        Parameters
        ----------
        probe_or_probegroup: Probe, list of Probe, or ProbeGroup
            The probe(s) to be attached to the recording
        group_mode: "auto" | "by_probe" | "by_shank" | "by_side", default: "auto"
            How to add the "group" property.
            "auto" is the best splitting possible that can be all at once when multiple probes, multiple shanks and two sides are present.
        in_place: bool
            False by default.
            Useful internally when extractor do self.set_probegroup(probe)
        channel_to_contact_ids : dict or None, default: None
            Optional mapping from channel_id to contact_id. If None, the mapping
            is inferred from device_channel_indices (legacy) or identity.

        Returns
        -------
        sub_recording: BaseRecording
            A view of the recording (ChannelSlice or clone or itself)
        """
        assert group_mode in (
            "auto",
            "by_probe",
            "by_shank",
            "by_side",
        ), "'group_mode' can be 'auto' 'by_probe' 'by_shank' or 'by_side'"

        # handle several input possibilities
        if isinstance(probe_or_probe_group, Probe):
            probegroup = ProbeGroup()
            probegroup.add_probe(probe_or_probe_group)
        elif isinstance(probe_or_probe_group, ProbeGroup):
            probegroup = probe_or_probe_group
        elif isinstance(probe_or_probe_group, list):
            assert all([isinstance(e, Probe) for e in probe_or_probe_group])
            probegroup = ProbeGroup()
            for probe in probe_or_probe_group:
                probegroup.add_probe(probe)
        else:
            raise ValueError("must give Probe or ProbeGroup or list of Probe")

        # Ensure all probes have contact_ids (assigns sequential string ids if missing)
        self._ensure_probes_have_contact_ids(probegroup)

        # check that the probes do not overlap
        num_probes = len(probegroup.probes)
        if num_probes > 1:
            check_probe_do_not_overlap(probegroup.probes)

        # total contacts across all probes
        total_num_contacts = sum(p.get_contact_count() for p in probegroup.probes)

        # Determine channel_to_contact_indices
        new_channel_ids = None  # only set by legacy path when channels need slicing
        if channel_to_contact_ids is not None:
            # Human-readable id mapping provided
            channel_to_contact_indices = self._build_channel_to_contact_indices_from_ids(
                probegroup, channel_to_contact_ids
            )
        elif all(probe.device_channel_indices is not None for probe in probegroup.probes):
            # Legacy path: read device_channel_indices from probes
            channel_to_contact_indices, new_channel_ids = self._build_channel_to_contact_indices_from_device_indices(
                probegroup
            )
        else:
            # Identity mapping: channel i reads from contact i
            channel_to_contact_indices = np.arange(self.get_num_channels(), dtype="int64")

        # Validate channel_to_contact_indices
        if channel_to_contact_indices.shape[0] != self.get_num_channels() and new_channel_ids is None:
            raise ValueError(
                f"Mapping length {channel_to_contact_indices.shape[0]} does not match "
                f"number of channels {self.get_num_channels()}."
            )
        if np.any(channel_to_contact_indices < 0):
            raise ValueError("All mapping indices must be >= 0.")
        if np.any(channel_to_contact_indices >= total_num_contacts):
            raise ValueError(
                f"Mapping contains index {channel_to_contact_indices.max()} but ProbeGroup only has "
                f"{total_num_contacts} contacts."
            )

        # Handle in_place / select_channels (legacy path may need channel slicing)
        if new_channel_ids is not None:
            # Legacy path: device_channel_indices may map a subset of channels
            if in_place:
                if not np.array_equal(new_channel_ids, self.get_channel_ids()):
                    raise Exception("set_probe(inplace=True) must have all channel indices")
                sub_recording = self
            else:
                if np.array_equal(new_channel_ids, self.get_channel_ids()):
                    sub_recording = self.clone()
                else:
                    sub_recording = self.select_channels(new_channel_ids)
        else:
            # New path: channel_to_contact_indices covers all channels, no slicing needed
            if in_place:
                sub_recording = self
            else:
                sub_recording = self.clone()

        # Build channel_to_contact_ids dict if not provided by the user
        if channel_to_contact_ids is None:
            all_contact_ids = np.concatenate([p.contact_ids for p in probegroup.probes])
            mapped_contact_ids = all_contact_ids[channel_to_contact_indices]
            channel_ids_for_map = new_channel_ids if new_channel_ids is not None else self.get_channel_ids()
            channel_to_contact_ids = dict(zip(channel_ids_for_map, mapped_contact_ids))

        # Store the probe and channel-to-contact mapping
        sub_recording._probe_group = probegroup
        sub_recording._channel_to_contact_indices = channel_to_contact_indices
        sub_recording._channel_to_contact_ids = channel_to_contact_ids

        # planar_contour is saved in annotations
        for probe_index, probe in enumerate(probegroup.probes):
            contour = probe.probe_planar_contour
            if contour is not None:
                sub_recording.set_annotation(f"probe_{probe_index}_planar_contour", contour, overwrite=True)

        # Compute and cache location property (always float64 for consistency)
        all_positions = np.concatenate([p.contact_positions for p in probegroup.probes])
        channel_positions = all_positions[channel_to_contact_indices]
        ndim = probegroup.ndim
        locations = np.asarray(channel_positions[:, :ndim], dtype="float64")
        sub_recording.set_property("location", locations, ids=None)

        # Compute and cache group property
        probe_as_numpy_array = self._compute_contact_vector_from(probegroup, channel_to_contact_indices)
        has_shank_id = "shank_ids" in probe_as_numpy_array.dtype.fields
        has_contact_side = "contact_sides" in probe_as_numpy_array.dtype.fields
        if group_mode == "auto":
            group_keys = ["probe_index"]
            if has_shank_id:
                group_keys += ["shank_ids"]
            if has_contact_side:
                group_keys += ["contact_sides"]
        elif group_mode == "by_probe":
            group_keys = ["probe_index"]
        elif group_mode == "by_shank":
            assert has_shank_id, "shank_ids is None in probe, you cannot group by shank"
            group_keys = ["probe_index", "shank_ids"]
        elif group_mode == "by_side":
            assert has_contact_side, "contact_sides is None in probe, you cannot group by side"
            if has_shank_id:
                group_keys = ["probe_index", "shank_ids", "contact_sides"]
            else:
                group_keys = ["probe_index", "contact_sides"]
        groups = np.zeros(probe_as_numpy_array.size, dtype="int64")
        unique_keys = np.unique(probe_as_numpy_array[group_keys])
        for group, a in enumerate(unique_keys):
            mask = np.ones(probe_as_numpy_array.size, dtype=bool)
            for k in group_keys:
                mask &= probe_as_numpy_array[k] == a[k]
            groups[mask] = group
        sub_recording.set_property("group", groups, ids=None)

        # add probe annotations to recording
        probes_info = []
        for probe in probegroup.probes:
            probes_info.append(probe.annotations)
        sub_recording.annotate(probes_info=probes_info)

        return sub_recording

    @staticmethod
    def _compute_contact_vector_from(probegroup, mapping):
        """Compute a contact_vector structured array from a probegroup and mapping.

        This is a static helper used during _set_probes() for group computation.
        """
        probe_as_numpy = probegroup.to_numpy(complete=True)
        probe_as_numpy = probe_as_numpy[mapping].copy()
        probe_as_numpy["device_channel_indices"] = np.arange(len(probe_as_numpy), dtype="int64")
        return probe_as_numpy

    def get_probe(self):
        probes = self.get_probes()
        assert len(probes) == 1, "there are several probe use .get_probes() or get_probegroup()"
        return probes[0]

    def get_probes(self):
        probegroup = self.get_probegroup()
        return probegroup.probes

    def get_probegroup(self):
        if self._probe_group is not None:
            probegroup = self._probe_group

            if "probes_info" in self.get_annotation_keys():
                probes_info = self.get_annotation("probes_info")
                for probe, probe_info in zip(probegroup.probes, probes_info):
                    probe.annotations = probe_info

            for probe_index, probe in enumerate(probegroup.probes):
                contour = self.get_annotation(f"probe_{probe_index}_planar_contour")
                if contour is not None:
                    probe.set_planar_contour(contour)

            return probegroup

        # Legacy fallback: reconstruct from contact_vector
        arr = super().get_property("contact_vector")
        if arr is None:
            positions = self.get_property("location")
            if positions is None:
                raise ValueError("There is no Probe attached to this recording. Use set_probe(...) to attach one.")
            else:
                warn("There is no Probe attached to this recording. Creating a dummy one with contact positions")
                probe = self.create_dummy_probe_from_locations(positions)
                probegroup = ProbeGroup()
                probegroup.add_probe(probe)
        else:
            probegroup = ProbeGroup.from_numpy(arr)

            if "probes_info" in self.get_annotation_keys():
                probes_info = self.get_annotation("probes_info")
                for probe, probe_info in zip(probegroup.probes, probes_info):
                    probe.annotations = probe_info

            for probe_index, probe in enumerate(probegroup.probes):
                contour = self.get_annotation(f"probe_{probe_index}_planar_contour")
                if contour is not None:
                    probe.set_planar_contour(contour)
        return probegroup

    def _extra_metadata_from_folder(self, folder):
        import json as json_module

        # load probe
        folder = Path(folder)
        if (folder / "probe.json").is_file():
            probegroup = read_probeinterface(folder / "probe.json")
            # Check for new-style id-based mapping
            mapping_file = folder / "channel_to_contact_ids.json"
            if mapping_file.is_file():
                with open(mapping_file) as f:
                    channel_to_contact_ids = json_module.load(f)
                self.set_probegroup(probegroup, in_place=True, channel_to_contact_ids=channel_to_contact_ids)
            elif (folder / "channel_to_contact_indices.npy").is_file():
                # Backward compat: old format saved indices as npy
                mapping = np.load(folder / "channel_to_contact_indices.npy")
                self._ensure_probes_have_contact_ids(probegroup)
                all_contact_ids = np.concatenate([p.contact_ids for p in probegroup.probes])
                mapped_ids = all_contact_ids[mapping]
                channel_to_contact_ids = dict(zip([str(ch) for ch in self.channel_ids], mapped_ids.tolist()))
                self.set_probegroup(probegroup, in_place=True, channel_to_contact_ids=channel_to_contact_ids)
            else:
                # Legacy: probegroup has device_channel_indices from probe.json
                self.set_probegroup(probegroup, in_place=True)

    def _extra_metadata_to_folder(self, folder):
        import json as json_module

        # save probe
        folder = Path(folder)
        if self._probe_group is not None:
            probegroup = self.get_probegroup()
            write_probeinterface(folder / "probe.json", probegroup)
            # Save the id-based mapping alongside probe.json
            channel_to_contact_ids = self.get_channel_to_contact_ids()
            # Convert keys to strings for JSON serialization
            serializable = {str(k): str(v) for k, v in channel_to_contact_ids.items()}
            with open(folder / "channel_to_contact_ids.json", "w") as f:
                json_module.dump(serializable, f)
        elif super().get_property("contact_vector") is not None:
            # Legacy fallback
            probegroup = self.get_probegroup()
            write_probeinterface(folder / "probe.json", probegroup)

    def create_dummy_probe_from_locations(self, locations, shape="circle", shape_params={"radius": 1}, axes="xy"):
        """
        Creates a "dummy" probe based on locations.

        Parameters
        ----------
        locations : np.array
            Array with channel locations (num_channels, ndim) [ndim can be 2 or 3]
        shape : str, default: "circle"
            Electrode shapes
        shape_params : dict, default: {"radius": 1}
            Shape parameters
        axes : str, default: "xy"
            If ndim is 3, indicates the axes that define the plane of the electrodes

        Returns
        -------
        probe : Probe
            The created probe
        """
        ndim = locations.shape[1]
        probe = Probe(ndim=2)
        if ndim == 3:
            locations_2d = select_axes(locations, axes)
        else:
            locations_2d = locations
        probe.set_contacts(locations_2d, shapes=shape, shape_params=shape_params)
        probe.set_contact_ids([str(i) for i in range(locations_2d.shape[0])])

        if ndim == 3:
            probe = probe.to_3d(axes=axes)

        return probe

    def set_dummy_probe_from_locations(self, locations, shape="circle", shape_params={"radius": 1}, axes="xy"):
        """
        Sets a "dummy" probe based on locations.

        Parameters
        ----------
        locations : np.array
            Array with channel locations (num_channels, ndim) [ndim can be 2 or 3]
        shape : str, default: default: "circle"
            Electrode shapes
        shape_params : dict, default: {"radius": 1}
            Shape parameters
        axes : "xy" | "yz" | "xz", default: "xy"
            If ndim is 3, indicates the axes that define the plane of the electrodes
        """
        probe = self.create_dummy_probe_from_locations(locations, shape=shape, shape_params=shape_params, axes=axes)
        self.set_probe(probe, in_place=True)

    def set_channel_locations(self, locations, channel_ids=None):
        if self.has_probe():
            raise ValueError("set_channel_locations(..) destroys the probe description, prefer _set_probes(..)")
        self.set_property("location", locations, ids=channel_ids)

    def get_channel_locations(self, channel_ids=None, axes: str = "xy") -> np.ndarray:
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        channel_indices = self.ids_to_indices(channel_ids)

        if self._probe_group is not None:
            # Fast path: use _probe_group + mapping directly
            all_positions = np.concatenate([p.contact_positions for p in self._probe_group.probes])
            channel_positions = all_positions[self._channel_to_contact_indices]
            ndim = len(axes)
            ndim_full = all_positions.shape[1]
            # Map axis names to column indices
            axis_map = {"x": 0, "y": 1, "z": 2}
            col_indices = [axis_map[a] for a in axes]
            positions = channel_positions[np.ix_(channel_indices, col_indices)]
            return np.asarray(positions, dtype="float64")

        # Fallback: location property (no probe set, locations added manually)
        locations = self.get_property("location")
        if locations is None:
            raise Exception("There are no channel locations")
        locations = np.asarray(locations)[channel_indices]
        return select_axes(locations, axes)

    def has_3d_locations(self) -> bool:
        return self.get_property("location").shape[1] == 3

    def clear_channel_locations(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channel()
        else:
            n = len(channel_ids)
        locations = np.zeros((n, 2)) * np.nan
        self.set_property("location", locations, ids=channel_ids)

    def set_channel_groups(self, groups, channel_ids=None):
        if "probes" in self._annotations:
            warn("set_channel_groups() destroys the probe description. Using set_probe() is preferable")
            self._annotations.pop("probes")
        self.set_property("group", groups, ids=channel_ids)

    def get_channel_groups(self, channel_ids=None):
        groups = self.get_property("group", ids=channel_ids)
        return groups

    def clear_channel_groups(self, channel_ids=None):
        if channel_ids is None:
            n = self.get_num_channels()
        else:
            n = len(channel_ids)
        groups = np.zeros(n, dtype="int64")
        self.set_property("group", groups, ids=channel_ids)

    def set_channel_gains(self, gains, channel_ids=None):
        if np.isscalar(gains):
            gains = [gains] * self.get_num_channels()
        self.set_property("gain_to_uV", gains, ids=channel_ids)

    def get_channel_gains(self, channel_ids=None):
        return self.get_property("gain_to_uV", ids=channel_ids)

    def set_channel_offsets(self, offsets, channel_ids=None):
        if np.isscalar(offsets):
            offsets = [offsets] * self.get_num_channels()
        self.set_property("offset_to_uV", offsets, ids=channel_ids)

    def get_channel_offsets(self, channel_ids=None):
        return self.get_property("offset_to_uV", ids=channel_ids)

    def get_channel_property(self, channel_id, key):
        values = self.get_property(key)
        v = values[self.id_to_index(channel_id)]
        return v

    def planarize(self, axes: str = "xy"):
        """
        Returns a Recording with a 2D probe from one with a 3D probe

        Parameters
        ----------
        axes : "xy" | "yz" |"xz", default: "xy"
            The axes to keep

        Returns
        -------
        BaseRecording
            The recording with 2D positions
        """
        assert self.has_3d_locations, "The 'planarize' function needs a recording with 3d locations"
        assert len(axes) == 2, "You need to specify 2 dimensions (e.g. 'xy', 'zy')"

        probe2d = self.get_probe().to_2d(axes=axes)
        recording2d = self.clone()
        recording2d.set_probe(probe2d, in_place=True)

        return recording2d

    def clone(self):
        """
        Clones an existing extractor into a new instance.
        Preserves _probe_group and _channel_to_contact_indices.
        """
        cloned = super().clone()
        if self._probe_group is not None:
            cloned._probe_group = self._probe_group
            cloned._channel_to_contact_indices = self._channel_to_contact_indices.copy()
            cloned._channel_to_contact_ids = (
                self._channel_to_contact_ids.copy() if self._channel_to_contact_ids is not None else None
            )
        return cloned

    def select_channels(self, channel_ids):
        """
        Returns a new object with sliced channels.

        Parameters
        ----------
        channel_ids : np.array or list
            The list of channels to keep

        Returns
        -------
        BaseRecordingSnippets
            The object with sliced channels
        """
        raise NotImplementedError

    def remove_channels(self, remove_channel_ids):
        """
        Returns a new object with removed channels.


        Parameters
        ----------
        remove_channel_ids : np.array or list
            The list of channels to remove

        Returns
        -------
        BaseRecordingSnippets
            The object with removed channels
        """
        return self._remove_channels(remove_channel_ids)

    def frame_slice(self, start_frame, end_frame):
        """
        Returns a new object with sliced frames.

        Parameters
        ----------
        start_frame : int
            The start frame
        end_frame : int
            The end frame

        Returns
        -------
        BaseRecordingSnippets
            The object with sliced frames
        """
        raise NotImplementedError

    def select_segments(self, segment_indices):
        """
        Return a new object with the segments specified by "segment_indices".

        Parameters
        ----------
        segment_indices : list of int
            List of segment indices to keep in the returned recording

        Returns
        -------
        BaseRecordingSnippets
            The onject with the selected segments
        """
        return self._select_segments(segment_indices)

    def split_by(self, property="group", outputs="dict"):
        """
        Splits object based on a certain property (e.g. "group")

        Parameters
        ----------
        property : str, default: "group"
            The property to use to split the object, default: "group"
        outputs : "dict" | "list", default: "dict"
            Whether to return a dict or a list

        Returns
        -------
        dict or list
            A dict or list with grouped objects based on property

        Raises
        ------
        ValueError
            Raised when property is not present
        """
        assert outputs in ("list", "dict")
        values = self.get_property(property)
        if values is None:
            raise ValueError(f"property {property} is not set")

        if outputs == "list":
            recordings = []
        elif outputs == "dict":
            recordings = {}
        for value in np.unique(values).tolist():
            (inds,) = np.nonzero(values == value)
            new_channel_ids = self.channel_ids[inds]
            subrec = self.select_channels(new_channel_ids)
            subrec.set_annotation("split_by_property", value=property)
            if outputs == "list":
                recordings.append(subrec)
            elif outputs == "dict":
                recordings[value] = subrec
        return recordings
