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

import math
from six import add_metaclass
from spinn_utilities.abstract_base import (
    AbstractBase, abstractproperty, abstractmethod)
from spinn_front_end_common.utilities.constants import (
    MICRO_TO_MILLISECOND_CONVERSION, MICRO_TO_SECOND_CONVERSION,
    BYTES_PER_WORD, BYTES_PER_SHORT)
from spynnaker.pyNN.exceptions import SynapticConfigurationException
from spynnaker.pyNN.models.neural_projections import (
    ProjectionApplicationEdge, ProjectionMachineEdge)


@add_metaclass(AbstractBase)
class AbstractSynapseDynamicsStructural(object):

    __slots__ = []

    # 7 32-bit numbers (fast; p_rew; s_max; app_no_atoms; machine_no_atoms;
    # low_atom; high_atom) + 2 4-word RNG seeds (shared_seed; local_seed)
    # + 1 32-bit number (no_pre_pops)
    _REWIRING_DATA_SIZE = (
        (7 * BYTES_PER_WORD) + (2 * 4 * BYTES_PER_WORD) + BYTES_PER_WORD)

    # Size excluding key_atom_info (as variable length)
    # 4 16-bit numbers (no_pre_vertices; sp_control; delay_lo; delay_hi)
    # + 3 32-bit numbers (weight; connection_type; total_no_atoms)
    _PRE_POP_INFO_BASE_SIZE = (4 * BYTES_PER_SHORT) + (3 * BYTES_PER_WORD)

    # 5 32-bit numbers (key; mask; n_atoms; lo_atom; m_pop_index)
    _KEY_ATOM_INFO_SIZE = (5 * BYTES_PER_WORD)

    # 1 16-bit number (neuron_index)
    # + 2 8-bit numbers (sub_pop_index; pop_index)
    _POST_TO_PRE_ENTRY_SIZE = BYTES_PER_SHORT + (2 * 1)

    #: Default value for frequency of rewiring
    DEFAULT_F_REW = 10**4
    #: Default value for initial weight on connection formation
    DEFAULT_INITIAL_WEIGHT = 0
    #: Default value for initial delay on connection formation
    DEFAULT_INITIAL_DELAY = 1
    #: Default value for maximum fan-in per target layer neuron
    DEFAULT_S_MAX = 32

    def __get_structural_edges(self, app_graph, app_vertex):
        """
        :param ~pacman.model.graphs.application.ApplicationGraph app_graph:
        :param ~pacman.model.graphs.application.ApplicationVertex app_vertex:
        :rtype: list(tuple(ProjectionApplicationEdge, SynapseInformation))
        """
        dynamics = None
        structural_edges = list()
        for app_edge in app_graph.get_edges_ending_at_vertex(app_vertex):
            if isinstance(app_edge, (ProjectionApplicationEdge, ProjectionMachineEdge)):
                for synapse_info in app_edge.synapse_information:
                    if isinstance(synapse_info.synapse_dynamics,
                                  AbstractSynapseDynamicsStructural):
                        if dynamics and dynamics != synapse_info.synapse_dynamics:
                            raise SynapticConfigurationException(
                                "Synapse dynamics must match exactly when "
                                "using multiple edges to the same population")
                        dynamics = synapse_info.synapse_dynamics
                        structural_edges.append(app_edge)
        return dynamics, structural_edges

    def get_structural_parameters_sdram_usage_in_bytes(
            self, application_graph, app_vertex, n_neurons):
        """ Get the size of the structural parameters

        :param ~pacman.model.graphs.application.ApplicationGraph \
                application_graph:
        :param ~spynnaker.pyNN.models.neuron.AbstractPopulationVertex \
                app_vertex:
        :param int n_neurons:
        :return: the size of the parameters, in bytes
        :rtype: int
        """
        # Work out how many sub-edges we will end up with, as this is used
        # for key_atom_info
        n_sub_edges = 0
        dynamics, structural_edges = self.__get_structural_edges(
            application_graph, app_vertex)
        # Also keep track of the parameter sizes
        param_sizes = self.partner_selection\
            .get_parameters_sdram_usage_in_bytes()
        for in_edge in structural_edges:
            if isinstance(in_edge, ProjectionMachineEdge):
                n_sub_edges += 1
            else:
                max_atoms = in_edge.pre_vertex.get_max_atoms_per_core()
                if in_edge.pre_vertex.n_atoms < max_atoms:
                    max_atoms = in_edge.pre_vertex.n_atoms
                n_sub_edges += int(math.ceil(
                    float(in_edge.pre_vertex.n_atoms) / float(max_atoms)))

        if dynamics:
            param_sizes += dynamics.formation\
                .get_parameters_sdram_usage_in_bytes()
            param_sizes += dynamics.elimination\
                .get_parameters_sdram_usage_in_bytes()

        return int((self._REWIRING_DATA_SIZE +
                   (self._PRE_POP_INFO_BASE_SIZE * len(structural_edges)) +
                   (self._KEY_ATOM_INFO_SIZE * n_sub_edges) +
                   (self._POST_TO_PRE_ENTRY_SIZE * n_neurons * self.s_max) +
                   param_sizes))

    @abstractmethod
    def write_structural_parameters(
            self, spec, region, machine_time_step, weight_scales,
            application_graph, app_vertex, post_slice,
            routing_info, synapse_indices):
        """ Write structural plasticity parameters

        :param ~data_specification.DataSpecificationGenerator spec:
        :param int region: region ID
        :param int machine_time_step:
        :param weight_scales:
        :type weight_scales: ~numpy.ndarray or list(float)
        :param ~pacman.model.graphs.application.ApplicationGraph\
                application_graph:
        :param AbstractPopulationVertex app_vertex:
        :param ~pacman.model.graphs.common.Slice post_slice:
        :param ~pacman.model.routing_info.RoutingInfo routing_info:
        :param dict(tuple(SynapseInformation,int),int) synapse_indices:
        """

    @abstractmethod
    def set_connections(
            self, connections, post_vertex_slice, app_edge, synapse_info,
            machine_edge):
        """ Set connections for structural plasticity

        :param ~numpy.ndarray connections:
        :param ~pacman.model.graphs.common.Slice post_vertex_slice:
        :param ProjectionApplicationEdge app_edge:
        :param SynapseInformation synapse_info:
        :param ProjectionMachineEdge machine_edge:
        """

    @abstractproperty
    def f_rew(self):
        """ The frequency of rewiring

        :rtype: float
        """

    @abstractproperty
    def s_max(self):
        """ The maximum number of synapses

        :rtype: int
        """

    @abstractproperty
    def seed(self):
        """ The seed to control the randomness
        """

    @abstractproperty
    def initial_weight(self):
        """ The weight of a formed connection

        :rtype: float
        """

    @abstractproperty
    def initial_delay(self):
        """ The delay of a formed connection

        :rtype: float
        """

    @abstractproperty
    def partner_selection(self):
        """ The partner selection rule

        :rtype: AbstractPartnerSelection
        """

    @abstractproperty
    def formation(self):
        """ The formation rule

        :rtype: AbstractFormation
        """

    @abstractproperty
    def elimination(self):
        """ The elimination rule

        :rtype: AbstractElimination
        """
