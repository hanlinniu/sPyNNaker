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


class SynapseInformation(object):
    """ Contains the synapse information including the connector, synapse type\
        and synapse dynamics
    """
    __slots__ = [
        "__connector",
        "__synapse_dynamics",
        "__synapse_type",
        "__weight",
        "__delay"]

    def __init__(self, connector, synapse_dynamics, synapse_type,
                 weight=None, delay=None):
        """
        :param connector:
        :type connector: AbstractConnector
        :param synapse_dynamics:
        :type synapse_dynamics: AbstractSynapseDynamics
        :param synapse_type:
        :type synapse_type: AbstractSynapseType
        :param weight:
        :type weight: int or None
        :param delay:
        :type delay: int or None
        """
        self.__connector = connector
        self.__synapse_dynamics = synapse_dynamics
        self.__synapse_type = synapse_type
        self.__weight = weight
        self.__delay = delay

    @property
    def connector(self):
        """
        :rtype: AbstractConnector
        """
        return self.__connector

    @property
    def synapse_dynamics(self):
        """
        :rtype: AbstractSynapseDynamics
        """
        return self.__synapse_dynamics

    @property
    def synapse_type(self):
        """
        :rtype: AbstractSynapseType
        """
        return self.__synapse_type

    @property
    def weight(self):
        """
        :rtype: int or None
        """
        return self.__weight

    @property
    def delay(self):
        """
        :rtype: int or None
        """
        return self.__delay
