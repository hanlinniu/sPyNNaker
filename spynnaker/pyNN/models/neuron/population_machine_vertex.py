# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from enum import Enum

from spinn_front_end_common.utilities import helpful_functions
from spinn_utilities.overrides import overrides
from pacman.model.graphs.machine import MachineVertex
from spinn_front_end_common.utilities.utility_objs import ProvenanceDataItem
from spinn_front_end_common.interface.provenance import (
    ProvidesProvenanceDataFromMachineImpl)
from spinn_front_end_common.interface.buffer_management.buffer_models import (
    AbstractReceiveBuffersToHost)
from spinn_front_end_common.utilities.helpful_functions import (
    locate_memory_region_for_placement)
from spinn_front_end_common.abstract_models import AbstractRecordable, \
    AbstractSupportsBitFieldGeneration, \
    AbstractSupportsBitFieldRoutingCompression
from spinn_front_end_common.interface.profiling import AbstractHasProfileData
from spinn_front_end_common.interface.profiling.profile_utils import (
    get_profiling_data)
from spynnaker.pyNN.utilities.constants import POPULATION_BASED_REGIONS


class PopulationMachineVertex(
        MachineVertex, AbstractReceiveBuffersToHost,
        ProvidesProvenanceDataFromMachineImpl, AbstractRecordable,
        AbstractHasProfileData, AbstractSupportsBitFieldGeneration,
        AbstractSupportsBitFieldRoutingCompression):

    __slots__ = [
        "__recorded_region_ids",
        "__resources",
        "__on_chip_generatable_area",
        "__on_chip_generatable_size"]

    class EXTRA_PROVENANCE_DATA_ENTRIES(Enum):
        """ Entries for the provenance data generated by standard neuron \
            models."""
        PRE_SYNAPTIC_EVENT_COUNT = 0
        SATURATION_COUNT = 1
        BUFFER_OVERFLOW_COUNT = 2
        CURRENT_TIMER_TIC = 3
        PLASTIC_SYNAPTIC_WEIGHT_SATURATION_COUNT = 4
        GHOST_POP_TABLE_SEARCHES = 5
        FAILED_TO_READ_BIT_FIELDS = 6
        DMA_COMPLETES = 7
        SPIKE_PROGRESSING_COUNT = 8
        INVALID_MASTER_POP_HITS = 9
        BIT_FIELD_FILTERED_COUNT = 10

    PROFILE_TAG_LABELS = {
        0: "TIMER",
        1: "DMA_READ",
        2: "INCOMING_SPIKE",
        3: "PROCESS_FIXED_SYNAPSES",
        4: "PROCESS_PLASTIC_SYNAPSES"}

    # x words needed for a bitfield covering 256 atoms
    WORDS_TO_COVER_256_ATOMS = 8

    # provenance data items
    BIT_FIELD_FILTERED_PACKETS = \
        "How many packets were filtered by the bitfield filterer."
    INVALID_MASTER_POP_HITS = "Invalid Master Pop hits"
    SPIKES_PROCESSED = "how many spikes were processed"
    DMA_COMPLETE = "DMA's that were completed"
    BIT_FIELDS_NOT_READ = "N bit fields not able to be read into DTCM"
    GHOST_SEARCHES = "Number of failed pop table searches"
    PLASTIC_WEIGHT_SATURATION = "Times_plastic_synaptic_weights_have_saturated"
    LAST_TIMER_TICK = "Last_timer_tic_the_core_ran_to"
    TOTAL_PRE_SYNAPTIC_EVENTS = "Total_pre_synaptic_events"
    LOST_INPUT_BUFFER_PACKETS = "Times_the_input_buffer_lost_packets"
    WEIGHT_SATURATION = "Times_synaptic_weights_have_saturated"

    N_ADDITIONAL_PROVENANCE_DATA_ITEMS = len(EXTRA_PROVENANCE_DATA_ENTRIES)

    def __init__(
            self, resources_required, recorded_region_ids, label, constraints):
        """
        :param resources_required:
        :param recorded_region_ids:
        :param label:
        :param constraints:
        :type sampling: bool
        """
        MachineVertex.__init__(self, label, constraints)
        AbstractSupportsBitFieldGeneration.__init__(self)
        AbstractRecordable.__init__(self)
        self.__recorded_region_ids = recorded_region_ids
        self.__resources = resources_required
        self.__on_chip_generatable_offset = None
        self.__on_chip_generatable_size = None

    def set_on_chip_generatable_area(self, offset, size):
        self.__on_chip_generatable_offset = offset
        self.__on_chip_generatable_size = size

    @overrides(AbstractSupportsBitFieldGeneration.bit_field_base_address)
    def bit_field_base_address(self, transceiver, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement=placement, transceiver=transceiver,
            region=POPULATION_BASED_REGIONS.BIT_FIELD_FILTER.value)

    @overrides(AbstractSupportsBitFieldRoutingCompression.
               key_to_atom_map_region_base_address)
    def key_to_atom_map_region_base_address(self, transceiver, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement=placement, transceiver=transceiver,
            region=POPULATION_BASED_REGIONS.BIT_FIELD_KEY_MAP.value)

    @overrides(AbstractSupportsBitFieldGeneration.bit_field_builder_region)
    def bit_field_builder_region(self, transceiver, placement):
        return helpful_functions.locate_memory_region_for_placement(
            placement=placement, transceiver=transceiver,
            region=POPULATION_BASED_REGIONS.BIT_FIELD_BUILDER.value)

    @overrides(AbstractSupportsBitFieldRoutingCompression.
               regeneratable_sdram_blocks_and_sizes)
    def regeneratable_sdram_blocks_and_sizes(
            self, transceiver, placement):
        synaptic_matrix_base_address = \
            helpful_functions.locate_memory_region_for_placement(
                placement=placement, transceiver=transceiver,
                region=POPULATION_BASED_REGIONS.SYNAPTIC_MATRIX.value)
        return [(
            self.__on_chip_generatable_offset + synaptic_matrix_base_address,
            self.__on_chip_generatable_size)]

    @property
    @overrides(MachineVertex.resources_required)
    def resources_required(self):
        return self.__resources

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
        return len(self.__recorded_region_ids) > 0

    @overrides(ProvidesProvenanceDataFromMachineImpl.
               get_provenance_data_from_machine)
    def get_provenance_data_from_machine(self, transceiver, placement):
        provenance_data = self._read_provenance_data(transceiver, placement)
        provenance_items = self._read_basic_provenance_items(
            provenance_data, placement)
        provenance_data = self._get_remaining_provenance_data_items(
            provenance_data)

        times_timer_tic_overran = 0
        for item in provenance_items:
            if item.names[-1] == self._TIMER_TICK_OVERRUN:
                times_timer_tic_overran = item.value

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
        n_ghost_searches = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.GHOST_POP_TABLE_SEARCHES.value]
        failed_to_read_bit_fields = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.FAILED_TO_READ_BIT_FIELDS.value]
        dma_completes = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.DMA_COMPLETES.value]
        spike_processing_count = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.SPIKE_PROGRESSING_COUNT.value]
        invalid_master_pop_hits = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.INVALID_MASTER_POP_HITS.value]
        n_packets_filtered_by_bit_field_filter = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.BIT_FIELD_FILTERED_COUNT.value]

        label, x, y, p, names = self._get_placement_details(placement)

        # translate into provenance data items
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.WEIGHT_SATURATION),
            n_saturations,
            report=n_saturations > 0,
            message=(
                "The weights from the synapses for {} on {}, {}, {} saturated "
                "{} times. If this causes issues you can increase the "
                "spikes_per_second and / or ring_buffer_sigma "
                "values located within the .spynnaker.cfg file.".format(
                    label, x, y, p, n_saturations))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.LOST_INPUT_BUFFER_PACKETS),
            n_buffer_overflows, report=n_buffer_overflows > 0,
            message=(
                "The input buffer for {} on {}, {}, {} lost packets on {} "
                "occasions. This is often a sign that the system is running "
                "too quickly for the number of neurons per core.  Please "
                "increase the timer_tic or time_scale_factor or decrease the "
                "number of neurons per core.".format(
                    label, x, y, p, n_buffer_overflows))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.TOTAL_PRE_SYNAPTIC_EVENTS),
            n_pre_synaptic_events))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.LAST_TIMER_TICK), last_timer_tick))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.PLASTIC_WEIGHT_SATURATION),
            n_plastic_saturations,
            report=n_plastic_saturations > 0,
            message=(
                "The weights from the plastic synapses for {} on {}, {}, {} "
                "saturated {} times. If this causes issue increase the "
                "spikes_per_second and / or ring_buffer_sigma values located "
                "within the .spynnaker.cfg file.".format(
                    label, x, y, p, n_plastic_saturations))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.GHOST_SEARCHES), n_ghost_searches,
            report=n_ghost_searches > 0,
            message=(
                "The number of failed population table searches for {} on {},"
                " {}, {} was {}. If this number is large relative to the "
                "predicted incoming spike rate, try increasing source and "
                "target neurons per core".format(
                    label, x, y, p, n_ghost_searches))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.BIT_FIELDS_NOT_READ),
            failed_to_read_bit_fields, report=failed_to_read_bit_fields > 0,
            message=(
                "The filter for stopping redundant DMA's couldn't be fully "
                "filled in, it failed to read {} entries, which means it "
                "required a max of {} extra bytes of DTCM (assuming cores "
                "have at max 255 neurons. Try reducing neurons per core, or "
                "size of buffers, or neuron params per neuron etc.".format(
                    failed_to_read_bit_fields,
                    failed_to_read_bit_fields *
                    self.WORDS_TO_COVER_256_ATOMS))))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.DMA_COMPLETE), dma_completes))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.SPIKES_PROCESSED),
            spike_processing_count))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.INVALID_MASTER_POP_HITS),
            invalid_master_pop_hits, report=invalid_master_pop_hits > 0,
            message=(
                "There were {} keys which were received by core {}:{}:{} which"
                " had no master pop entry for it. This is a error, which most "
                "likely strives from bad routing.".format(
                    invalid_master_pop_hits, x, y, p))))
        provenance_items.append((ProvenanceDataItem(
            self._add_name(names, self.BIT_FIELD_FILTERED_PACKETS),
            n_packets_filtered_by_bit_field_filter,
            report=(
                n_packets_filtered_by_bit_field_filter > 0 and (
                    n_buffer_overflows > 0 or times_timer_tic_overran > 0)),
            message=(
                "There were {} packets received by {}:{}:{} that were "
                "filtered by the Bitfield filterer on the core. These packets "
                "were having to be stored and processed on core, which means "
                "the core may not be running as efficiently as it could. "
                "Please adjust the network or the mapping so that these "
                "packets are filtered in the router to improve "
                "performance.".format(
                    n_packets_filtered_by_bit_field_filter, x, y, p)))))

        return provenance_items

    @overrides(AbstractReceiveBuffersToHost.get_recorded_region_ids)
    def get_recorded_region_ids(self):
        return self.__recorded_region_ids

    @overrides(AbstractReceiveBuffersToHost.get_recording_region_base_address)
    def get_recording_region_base_address(self, txrx, placement):
        return locate_memory_region_for_placement(
            placement, POPULATION_BASED_REGIONS.NEURON_RECORDING.value, txrx)

    @overrides(AbstractHasProfileData.get_profile_data)
    def get_profile_data(self, transceiver, placement):
        return get_profiling_data(
            POPULATION_BASED_REGIONS.PROFILING.value,
            self.PROFILE_TAG_LABELS, transceiver, placement)
