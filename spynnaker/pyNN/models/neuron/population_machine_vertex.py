from spinn_utilities.overrides import overrides

# pacman imports
from pacman.model.graphs.machine import MachineVertex

# spinn front end common imports
from spinn_front_end_common.utilities.utility_objs import ProvenanceDataItem
from spinn_front_end_common.interface.provenance \
    import ProvidesProvenanceDataFromMachineImpl
from spinn_front_end_common.interface.buffer_management.buffer_models \
    import AbstractReceiveBuffersToHost
from spinn_front_end_common.interface.buffer_management\
    import recording_utilities
from spinn_front_end_common.utilities.helpful_functions \
    import locate_memory_region_for_placement
from spinn_front_end_common.abstract_models import AbstractRecordable
from spinn_front_end_common.interface.profiling import AbstractHasProfileData
from spinn_front_end_common.interface.profiling.profile_utils \
    import get_profiling_data

# spynnaker imports
from spynnaker.pyNN.utilities.constants import POPULATION_BASED_REGIONS

from enum import Enum


class PopulationMachineVertex(
        MachineVertex, AbstractReceiveBuffersToHost,
        ProvidesProvenanceDataFromMachineImpl, AbstractRecordable,
        AbstractHasProfileData):
    __slots__ = [
        "_buffered_sdram_per_timestep",
        "_is_recording",
        "_minimum_buffer_sdram_usage",
        "_overflow_sdram",
        "_resources"]

    # entries for the provenance data generated by standard neuron models
    EXTRA_PROVENANCE_DATA_ENTRIES = Enum(
        value="EXTRA_PROVENANCE_DATA_ENTRIES",
        names=[("PRE_SYNAPTIC_EVENT_COUNT", 0),
               ("SATURATION_COUNT", 1),
               ("BUFFER_OVERFLOW_COUNT", 2),
               ("CURRENT_TIMER_TIC", 3),
               ("PLASTIC_SYNAPTIC_WEIGHT_SATURATION_COUNT", 4),
               ("MAX_FLUSHED_SPIKES", 5),
               ("TOTAL_FLUSHED_SPIKES", 6)])

    PROFILE_TAG_LABELS = {
        0: "TIMER",
        1: "DMA_READ",
        2: "INCOMING_SPIKE",
        3: "PROCESS_FIXED_SYNAPSES",
        4: "PROCESS_PLASTIC_SYNAPSES"}

    N_ADDITIONAL_PROVENANCE_DATA_ITEMS = len(EXTRA_PROVENANCE_DATA_ENTRIES)

    def __init__(
            self, resources_required, is_recording, minimum_buffer_sdram_usage,
            buffered_sdram_per_timestep, label, constraints=None,
            overflow_sdram=0):
        """
        :param resources_required:
        :param is_recording:
        :param minimum_buffer_sdram_usage:
        :param buffered_sdram_per_timestep:
        :param label:
        :param constraints:
        :param overflow_sdram: Extra SDRAM that may be required if\
            buffered_sdram_per_timestep is an average
        :type sampling: bool
        """
        MachineVertex.__init__(self, label, constraints)
        AbstractRecordable.__init__(self)
        self._is_recording = is_recording
        self._resources = resources_required
        self._minimum_buffer_sdram_usage = minimum_buffer_sdram_usage
        self._buffered_sdram_per_timestep = buffered_sdram_per_timestep
        self._overflow_sdram = overflow_sdram

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return self._resources

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._provenance_region_id)
    def _provenance_region_id(self):
        return POPULATION_BASED_REGIONS.PROVENANCE_DATA.value

    @property
    @overrides(ProvidesProvenanceDataFromMachineImpl._n_additional_data_items)
    def _n_additional_data_items(self):
        return self.N_ADDITIONAL_PROVENANCE_DATA_ITEMS

    @overrides(AbstractRecordable.is_recording)
    def is_recording(self):
        return self._is_recording

    @overrides(ProvidesProvenanceDataFromMachineImpl.
               get_provenance_data_from_machine)
    def get_provenance_data_from_machine(self, transceiver, placement):
        provenance_data = self._read_provenance_data(transceiver, placement)
        provenance_items = self._read_basic_provenance_items(
            provenance_data, placement)
        provenance_data = self._get_remaining_provenance_data_items(
            provenance_data)

        n_saturations = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.SATURATION_COUNT.value]
        n_buffer_overflows = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.BUFFER_OVERFLOW_COUNT.value]
        n_pre_synaptic_events = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.PRE_SYNAPTIC_EVENT_COUNT.value]
        last_timer_tick = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.CURRENT_TIMER_TIC.value]
        n_plastic_saturations = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.
            PLASTIC_SYNAPTIC_WEIGHT_SATURATION_COUNT.value]
        max_unprocessed_spikes = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.MAX_FLUSHED_SPIKES.value]
        total_unprocessed_spikes = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.TOTAL_FLUSHED_SPIKES.value]

        label, x, y, p, names = self._get_placement_details(placement)

        # translate into provenance data items
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "Times_synaptic_weights_have_saturated"),
            n_saturations,
            report=n_saturations > 0,
            message=(
                "The weights from the synapses for {} on {}, {}, {} saturated "
                "{} times. If this causes issues you can increase the "
                "spikes_per_second and / or ring_buffer_sigma "
                "values located within the .spynnaker.cfg file.".format(
                    label, x, y, p, n_saturations))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "Times_the_input_buffer_lost_packets"),
            n_buffer_overflows,
            report=n_buffer_overflows > 0,
            message=(
                "The input buffer for {} on {}, {}, {} lost packets on {} "
                "occasions. This is often a sign that the system is running "
                "too quickly for the number of neurons per core.  Please "
                "increase the timer_tic or time_scale_factor or decrease the "
                "number of neurons per core.".format(
                    label, x, y, p, n_buffer_overflows))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "Total_pre_synaptic_events"),
            n_pre_synaptic_events))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, "Last_timer_tic_the_core_ran_to"),
            last_timer_tick))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names,
                           "Times_plastic_synaptic_weights_have_saturated"),
            n_plastic_saturations,
            report=n_plastic_saturations > 0,
            message=(
                "The weights from the plastic synapses for {} on {}, {}, {} "
                "saturated {} times. If this causes issue increase the "
                "spikes_per_second and / or ring_buffer_sigma values located "
                "within the .spynnaker.cfg file.".format(
                    label, x, y, p, n_plastic_saturations))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names,
                           "Max unprocessed spikes in a single time step"),
            max_unprocessed_spikes,
            report=max_unprocessed_spikes > 0,
            message=(
                "Max unprocessed spikes in a timestep for {} on {}, {}, {} "
                "was {}. Please adjust the timescale factor".format(
                    label, x, y, p, max_unprocessed_spikes))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names,
                           "Total unprocessed spikes over entrie simulation"),
            total_unprocessed_spikes,
            report=total_unprocessed_spikes > 0,
            message=(
                "Total unprocessed spikes over entire simulation for {} on"
                "{}, {}, {} was {}. Please adjust the timescale factor".format(
                    label, x, y, p, total_unprocessed_spikes))))

        return provenance_items

    @overrides(AbstractReceiveBuffersToHost.get_minimum_buffer_sdram_usage)
    def get_minimum_buffer_sdram_usage(self):
        return sum(self._minimum_buffer_sdram_usage)

    @overrides(AbstractReceiveBuffersToHost.get_n_timesteps_in_buffer_space)
    def get_n_timesteps_in_buffer_space(self, buffer_space, machine_time_step):
        safe_space = buffer_space - self._overflow_sdram
        return recording_utilities.get_n_timesteps_in_buffer_space(
            safe_space, self._buffered_sdram_per_timestep)

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return recording_utilities.get_recorded_region_ids(
            self._buffered_sdram_per_timestep)

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, txrx, placement):
        return locate_memory_region_for_placement(
            placement, POPULATION_BASED_REGIONS.RECORDING.value, txrx)

    @overrides(AbstractHasProfileData.get_profile_data)
    def get_profile_data(self, transceiver, placement):
        return get_profiling_data(
            POPULATION_BASED_REGIONS.PROFILING.value,
            self.PROFILE_TAG_LABELS, transceiver, placement)
