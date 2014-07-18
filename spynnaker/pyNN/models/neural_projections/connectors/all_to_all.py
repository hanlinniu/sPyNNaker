from spynnaker.pyNN.models.neural_projections.connectors.abstract_connector \
    import AbstractConnector
from spynnaker.pyNN.models.neural_properties.synaptic_list import SynapticList
from spynnaker.pyNN.models.neural_properties.synapse_row_info \
    import SynapseRowInfo
from spynnaker.pyNN.models.neural_properties.randomDistributions \
    import generateParameterArray
import numpy


class AllToAllConnector(AbstractConnector):
    """
    Connects all cells in the presynaptic pynn_population.py to all cells in the
    postsynaptic pynn_population.py.

    :param `bool` allow_self_connections: 
        if the connector is used to connect a
        Population to itself, this flag determines whether a neuron is
        allowed to connect to itself, or only to other neurons in the
        Population.
    :param `float` weights:
        may either be a float, a !RandomDistribution object, a list/
        1D array with at least as many items as connections to be
        created. Units nA.
    :param `float` delays:  -- as `weights`. If `None`, all synaptic delays will 
        be set to the global minimum delay.
    :param `pyNN.Space` space: 
        a Space object, needed if you wish to specify distance-
        dependent weights or delays - not implemented
        
    """
    def __init__(self, weights=0.0, delays=1, allow_self_connections=True):
        """
        Creates a new AllToAllConnector.
        """
        self._weights = weights
        self._delays = delays
        self._allow_self_connections = allow_self_connections
        
    def generate_synapse_list(self, prevertex, postvertex, delay_scale, 
                              synapse_type):
        
        connection_list = list()
        for _ in range(0, prevertex.atoms):
            present = numpy.ones(postvertex.atoms, dtype=numpy.uint32)
            n_present = postvertex.atoms
            
            ids = numpy.where(present)[0]
            delays = (generateParameterArray(self._delays, n_present, present)
                      * delay_scale)
            weights = generateParameterArray(self._weights, n_present, present)
            synapse_types = (numpy.ones(len(ids), dtype='uint32')
                             * synapse_type)
            
            connection_list.append(SynapseRowInfo(ids, weights, delays,
                                   synapse_types))
                    
        return SynapticList(connection_list)
