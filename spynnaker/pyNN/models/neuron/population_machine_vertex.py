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
from spinn_utilities.overrides import overrides
from pacman.model.graphs.machine import MachineVertex
from spinn_front_end_common.utilities.utility_objs import ProvenanceDataItem
from spinn_front_end_common.interface.provenance import (
    ProvidesProvenanceDataFromMachineImpl)
from spinn_front_end_common.interface.buffer_management.buffer_models import (
    AbstractReceiveBuffersToHost)
from spinn_front_end_common.utilities.helpful_functions import (
    locate_memory_region_for_placement)
from spinn_front_end_common.abstract_models import AbstractRecordable
from spinn_front_end_common.interface.profiling import AbstractHasProfileData
from spinn_front_end_common.interface.profiling.profile_utils import (
    get_profiling_data)
from spynnaker.pyNN.utilities.constants import POPULATION_BASED_REGIONS


class PopulationMachineVertex(
        MachineVertex, AbstractReceiveBuffersToHost,
        ProvidesProvenanceDataFromMachineImpl, AbstractRecordable,
        AbstractHasProfileData):

    __slots__ = [
        "__recorded_region_ids",
        "__resources",
        "__drop_late_spikes"]

    class EXTRA_PROVENANCE_DATA_ENTRIES(Enum):
        """ Entries for the provenance data generated by standard neuron \
            models."""
        PRE_SYNAPTIC_EVENT_COUNT = 0
        SATURATION_COUNT = 1
        BUFFER_OVERFLOW_COUNT = 2
        CURRENT_TIMER_TIC = 3
        PLASTIC_SYNAPTIC_WEIGHT_SATURATION_COUNT = 4
        N_REWIRES = 5
        # the number of packets that were dropped as they arrived too late
        # to be processed
        N_LATE_SPIKES = 6

    SATURATION_COUNT_NAME = "Times_synaptic_weights_have_saturated"
    SATURATION_COUNT_MESSAGE = (
        "The weights from the synapses for {} on {}, {}, {} saturated "
        "{} times. If this causes issues you can increase the "
        "spikes_per_second and / or ring_buffer_sigma "
        "values located within the .spynnaker.cfg file.")

    INPUT_BUFFER_FULL_NAME = "Times_the_input_buffer_lost_packets"
    INPUT_BUFFER_FULL_MESSAGE = (
        "The input buffer for {} on {}, {}, {} lost packets on {} "
        "occasions. This is often a sign that the system is running "
        "too quickly for the number of neurons per core.  Please "
        "increase the timer_tic or time_scale_factor or decrease the "
        "number of neurons per core.")

    TOTAL_PRE_SYNAPTIC_EVENT_NAME = "Total_pre_synaptic_events"
    LAST_TIMER_TICK_NAME = "Last_timer_tic_the_core_ran_to"
    N_RE_WIRES_NAME = "Number_of_rewires"

    SATURATED_PLASTIC_WEIGHTS_NAME = (
        "Times_plastic_synaptic_weights_have_saturated")
    SATURATED_PLASTIC_WEIGHTS_MESSAGE = (
        "The weights from the plastic synapses for {} on {}, {}, {} "
        "saturated {} times. If this causes issue increase the "
        "spikes_per_second and / or ring_buffer_sigma values located "
        "within the .spynnaker.cfg file.")

    _N_LATE_SPIKES_NAME = "Number_of_late_spikes"
    _N_LATE_SPIKES_MESSAGE_DROP = (
        "{} packets from {} on {}, {}, {} were dropped from the input buffer, "
        "because they arrived too late to be processed in a given time step. "
        "Try increasing the time_scale_factor located within the "
        ".spynnaker.cfg file or in the pynn.setup() method.")
    _N_LATE_SPIKES_MESSAGE_NO_DROP = (
        "{} packets from {} on {}, {}, {} arrived too late to be processed in"
        " a given time step. "
        "Try increasing the time_scale_factor located within the "
        ".spynnaker.cfg file or in the pynn.setup() method.")

    PROFILE_TAG_LABELS = {
        0: "TIMER",
        1: "DMA_READ",
        2: "INCOMING_SPIKE",
        3: "PROCESS_FIXED_SYNAPSES",
        4: "PROCESS_PLASTIC_SYNAPSES"}

    N_ADDITIONAL_PROVENANCE_DATA_ITEMS = len(EXTRA_PROVENANCE_DATA_ENTRIES)

    def __init__(
            self, resources_required, recorded_region_ids, label, constraints,
            drop_late_spikes):
        """
        :param resources_required:
        :param recorded_region_ids:
        :param label:
        :param constraints:
        :param drop_late_spikes:
        """
        MachineVertex.__init__(self, label, constraints)
        AbstractRecordable.__init__(self)
        self.__recorded_region_ids = recorded_region_ids
        self.__resources = resources_required
        self.__drop_late_spikes = drop_late_spikes

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
        n_rewires = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.N_REWIRES.value]
        n_late_packets = provenance_data[
            self.EXTRA_PROVENANCE_DATA_ENTRIES.N_LATE_SPIKES.value]

        label, x, y, p, names = self._get_placement_details(placement)

        # translate into provenance data items
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.SATURATION_COUNT_NAME),
            n_saturations, report=n_saturations > 0,
            message=self.SATURATION_COUNT_MESSAGE.format(
                label, x, y, p, n_saturations)))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.INPUT_BUFFER_FULL_NAME),
            n_buffer_overflows, report=n_buffer_overflows > 0,
            message=self.INPUT_BUFFER_FULL_MESSAGE.format(
                label, x, y, p, n_buffer_overflows)))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.TOTAL_PRE_SYNAPTIC_EVENT_NAME),
            n_pre_synaptic_events))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.LAST_TIMER_TICK_NAME),
            last_timer_tick))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.SATURATED_PLASTIC_WEIGHTS_NAME),
            n_plastic_saturations, report=n_plastic_saturations > 0,
            message=self.SATURATED_PLASTIC_WEIGHTS_MESSAGE.format(
                label, x, y, p, n_plastic_saturations)))
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self.N_RE_WIRES_NAME), n_rewires))

        late_message = (
            self._N_LATE_SPIKES_MESSAGE_DROP if self.__drop_late_spikes
            else self._N_LATE_SPIKES_MESSAGE_NO_DROP)
        provenance_items.append(ProvenanceDataItem(
            self._add_name(names, self._N_LATE_SPIKES_NAME),
            n_late_packets, report=n_late_packets > 0,
            message=late_message.format(n_late_packets, label, x, y, p)))

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
