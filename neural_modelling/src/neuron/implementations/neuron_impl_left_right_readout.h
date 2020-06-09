#ifndef _NEURON_IMPL_LEFT_RIGHT_READOUT_H_
#define _NEURON_IMPL_LEFT_RIGHT_READOUT_H_

#include "neuron_impl.h"

// Includes for model parts used in this implementation
#include <neuron/synapse_types/synapse_type_eprop_adaptive.h>
#include <neuron/models/neuron_model_left_right_readout_impl.h>
#include <neuron/input_types/input_type_current.h>
#include <neuron/additional_inputs/additional_input_none_impl.h>
#include <neuron/threshold_types/threshold_type_static.h>

// Further includes
#include <common/out_spikes.h>
#include <common/maths-util.h>
#include <recording.h>
#include <debug.h>
#include <random.h>
#include <log.h>

#define V_RECORDING_INDEX 0
#define GSYN_EXCITATORY_RECORDING_INDEX 1
#define GSYN_INHIBITORY_RECORDING_INDEX 2

#ifndef NUM_EXCITATORY_RECEPTORS
#define NUM_EXCITATORY_RECEPTORS 1
#error NUM_EXCITATORY_RECEPTORS was undefined.  It should be defined by a synapse\
       shaping include
#endif

#ifndef NUM_INHIBITORY_RECEPTORS
#define NUM_INHIBITORY_RECEPTORS 1
#error NUM_INHIBITORY_RECEPTORS was undefined.  It should be defined by a synapse\
       shaping include
#endif

//! Array of neuron states
neuron_pointer_t neuron_array;

//! Input states array
static input_type_pointer_t input_type_array;

//! Additional input array
static additional_input_pointer_t additional_input_array;

//! Threshold states array
static threshold_type_pointer_t threshold_type_array;

//! Global parameters for the neurons
static global_neuron_params_pointer_t global_parameters;

// The synapse shaping parameters
static synapse_param_t *neuron_synapse_shaping_params;

static REAL next_spike_time = 0;
extern uint32_t time;
extern key_t key;
extern REAL learning_signal;
static uint32_t target_ind = 0;

// recording prams
uint32_t is_it_right = 0;
//uint32_t choice = 0;

// Left right parameters
typedef enum
{
    STATE_CUE,
    STATE_WAITING,
    STATE_PROMPT,
} current_state_t;

current_state_t current_state = 0;
uint32_t current_time = 0;
uint32_t cue_number = 0;
uint32_t total_cues = 1;
uint32_t current_cue_direction = 2; // 0 = left, 1 = right
uint32_t accumulative_direction = 0; // if > total_cues / 2 = right
uint32_t wait_between_cues = 50; // ms
uint32_t duration_of_cue = 100; // ms
uint32_t wait_before_result = 1000; // ms but should be a range between 500-1500
uint32_t prompt_duration = 150; //ms
//uint32_t ticks_for_mean = 0;
bool start_prompt = false;
accum softmax_0 = 0k;
accum softmax_1 = 0k;
//REAL payload;
bool completed_broadcast = true;


static bool neuron_impl_initialise(uint32_t n_neurons) {

    // allocate DTCM for the global parameter details
    if (sizeof(global_neuron_params_t) > 0) {
        global_parameters = (global_neuron_params_t *) spin1_malloc(
            sizeof(global_neuron_params_t));
        if (global_parameters == NULL) {
            log_error("Unable to allocate global neuron parameters"
                      "- Out of DTCM");
            return false;
        }
    }

    // Allocate DTCM for neuron array
    if (sizeof(neuron_t) != 0) {
        neuron_array = (neuron_t *) spin1_malloc(n_neurons * sizeof(neuron_t));
        if (neuron_array == NULL) {
            log_error("Unable to allocate neuron array - Out of DTCM");
            return false;
        }
    }

    // Allocate DTCM for input type array and copy block of data
    if (sizeof(input_type_t) != 0) {
        input_type_array = (input_type_t *) spin1_malloc(
            n_neurons * sizeof(input_type_t));
        if (input_type_array == NULL) {
            log_error("Unable to allocate input type array - Out of DTCM");
            return false;
        }
    }

    // Allocate DTCM for additional input array and copy block of data
    if (sizeof(additional_input_t) != 0) {
        additional_input_array = (additional_input_pointer_t) spin1_malloc(
            n_neurons * sizeof(additional_input_t));
        if (additional_input_array == NULL) {
            log_error("Unable to allocate additional input array"
                      " - Out of DTCM");
            return false;
        }
    }

    // Allocate DTCM for threshold type array and copy block of data
    if (sizeof(threshold_type_t) != 0) {
        threshold_type_array = (threshold_type_t *) spin1_malloc(
            n_neurons * sizeof(threshold_type_t));
        if (threshold_type_array == NULL) {
            log_error("Unable to allocate threshold type array - Out of DTCM");
            return false;
        }
    }

    // Allocate DTCM for synapse shaping parameters
    if (sizeof(synapse_param_t) != 0) {
        neuron_synapse_shaping_params = (synapse_param_t *) spin1_malloc(
            n_neurons * sizeof(synapse_param_t));
        if (neuron_synapse_shaping_params == NULL) {
            log_error("Unable to allocate synapse parameters array"
                " - Out of DTCM");
            return false;
        }
    }

    // Seed the random input
    validate_mars_kiss64_seed(global_parameters->kiss_seed);

    // Initialise pointers to Neuron parameters in STDP code
//    synapse_dynamics_set_neuron_array(neuron_array);
    log_info("set pointer to neuron array in stdp code");

    return true;
}

static void neuron_impl_add_inputs(
        index_t synapse_type_index, index_t neuron_index,
        input_t weights_this_timestep) {
    // simple wrapper to synapse type input function
    synapse_param_pointer_t parameters =
            &(neuron_synapse_shaping_params[neuron_index]);
    synapse_types_add_neuron_input(synapse_type_index,
            parameters, weights_this_timestep);
}

static void neuron_impl_load_neuron_parameters(
        address_t address, uint32_t next, uint32_t n_neurons) {
    log_debug("reading parameters, next is %u, n_neurons is %u ",
        next, n_neurons);

    //log_debug("writing neuron global parameters");
    spin1_memcpy(global_parameters, &address[next],
            sizeof(global_neuron_params_t));
    next += (sizeof(global_neuron_params_t) + 3) / 4;

    log_debug("reading neuron local parameters");
    spin1_memcpy(neuron_array, &address[next], n_neurons * sizeof(neuron_t));
    next += ((n_neurons * sizeof(neuron_t)) + 3) / 4;

    log_debug("reading input type parameters");
    spin1_memcpy(input_type_array, &address[next],
            n_neurons * sizeof(input_type_t));
    next += ((n_neurons * sizeof(input_type_t)) + 3) / 4;

    log_debug("reading threshold type parameters");
    spin1_memcpy(threshold_type_array, &address[next],
           n_neurons * sizeof(threshold_type_t));
    next += ((n_neurons * sizeof(threshold_type_t)) + 3) / 4;

    log_debug("reading synapse parameters");
    spin1_memcpy(neuron_synapse_shaping_params, &address[next],
           n_neurons * sizeof(synapse_param_t));
    next += ((n_neurons * sizeof(synapse_param_t)) + 3) / 4;

    log_debug("reading additional input type parameters");
        spin1_memcpy(additional_input_array, &address[next],
               n_neurons * sizeof(additional_input_t));
    next += ((n_neurons * sizeof(additional_input_t)) + 3) / 4;

    neuron_model_set_global_neuron_params(global_parameters);

    io_printf(IO_BUF, "\nPrinting global params\n");
    io_printf(IO_BUF, "seed 1: %u \n", global_parameters->kiss_seed[0]);
    io_printf(IO_BUF, "seed 2: %u \n", global_parameters->kiss_seed[1]);
    io_printf(IO_BUF, "seed 3: %u \n", global_parameters->kiss_seed[2]);
    io_printf(IO_BUF, "seed 4: %u \n", global_parameters->kiss_seed[3]);
    io_printf(IO_BUF, "ticks_per_second: %k \n\n", global_parameters->ticks_per_second);
//    io_printf(IO_BUF, "prob_command: %k \n\n", global_parameters->prob_command);
    io_printf(IO_BUF, "rate on: %k \n\n", global_parameters->rate_on);
    io_printf(IO_BUF, "rate off: %k \n\n", global_parameters->rate_off);
    io_printf(IO_BUF, "mean 0: %k \n\n", global_parameters->mean_0);
    io_printf(IO_BUF, "mean 1: %k \n\n", global_parameters->mean_1);
    io_printf(IO_BUF, "poisson key: %u \n\n", global_parameters->p_key);
    io_printf(IO_BUF, "poisson pop size: %u \n\n", global_parameters->p_pop_size);


    for (index_t n = 0; n < n_neurons; n++) {
        neuron_model_print_parameters(&neuron_array[n]);
    }

//    io_printf(IO_BUF, "size of global params: %u",
//    		sizeof(global_neuron_params_t));



    #if LOG_LEVEL >= LOG_DEBUG
        log_debug("-------------------------------------\n");
        for (index_t n = 0; n < n_neurons; n++) {
            neuron_model_print_parameters(&neuron_array[n]);
        }
        log_debug("-------------------------------------\n");
        //}
    #endif // LOG_LEVEL >= LOG_DEBUG
}




static bool neuron_impl_do_timestep_update(index_t neuron_index,
        input_t external_bias, state_t *recorded_variable_values) {

    // Get the neuron itself
    neuron_pointer_t neuron = &neuron_array[neuron_index];
    bool spike = false;

//    current_time = time & 0x3ff; // repeats on a cycle of 1024 entries in array

//    io_printf(IO_BUF, "Updating Neuron Index: %u\n", neuron_index);
//    io_printf(IO_BUF, "Target: %k\n\n",
//    		global_parameters->target_V[target_ind]);

    // Get the input_type parameters and voltage for this neuron
    input_type_pointer_t input_type = &input_type_array[neuron_index];

    // Get threshold and additional input parameters for this neuron
    threshold_type_pointer_t threshold_type =
    		&threshold_type_array[neuron_index];
    additional_input_pointer_t additional_input =
    		&additional_input_array[neuron_index];
    synapse_param_pointer_t synapse_type =
    		&neuron_synapse_shaping_params[neuron_index];

    // Get the voltage
    state_t voltage = neuron_model_get_membrane_voltage(neuron);


    // Get the exc and inh values from the synapses
    input_t* exc_value = synapse_types_get_excitatory_input(synapse_type);
    input_t* inh_value = synapse_types_get_inhibitory_input(synapse_type);

    // Call functions to obtain exc_input and inh_input
    input_t* exc_input_values = input_type_get_input_value(
           exc_value, input_type, NUM_EXCITATORY_RECEPTORS);
    input_t* inh_input_values = input_type_get_input_value(
           inh_value, input_type, NUM_INHIBITORY_RECEPTORS);

    // Sum g_syn contributions from all receptors for recording
//    REAL total_exc = 0;
//    REAL total_inh = 0;
//
//    for (int i = 0; i < NUM_EXCITATORY_RECEPTORS-1; i++){
//    	total_exc += exc_input_values[i];
//    }
//    for (int i = 0; i < NUM_INHIBITORY_RECEPTORS-1; i++){
//    	total_inh += inh_input_values[i];
//    }

    // Call functions to get the input values to be recorded
//    recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = total_exc;
//    recorded_variable_values[GSYN_INHIBITORY_RECORDING_INDEX] = total_inh;

    // Call functions to convert exc_input and inh_input to current
    input_type_convert_excitatory_input_to_current(
    		exc_input_values, input_type, voltage);
    input_type_convert_inhibitory_input_to_current(
    		inh_input_values, input_type, voltage);

    external_bias += additional_input_get_input_value_as_current(
    		additional_input, voltage);

    if (neuron_index == 0){
        // update neuron parameters
        state_t result = neuron_model_state_update(
                NUM_EXCITATORY_RECEPTORS, exc_input_values,
                NUM_INHIBITORY_RECEPTORS, inh_input_values,
                external_bias, neuron, -50k);
        // Finally, set global membrane potential to updated value
        global_parameters->readout_V_0 = result;

    } else if (neuron_index == 1){
        // update neuron parameters
        learning_signal *= -1.k;
        state_t result = neuron_model_state_update(
                NUM_EXCITATORY_RECEPTORS, exc_input_values,
                NUM_INHIBITORY_RECEPTORS, inh_input_values,
                external_bias, neuron, -50k);
        learning_signal *= -1.k;
        // Finally, set global membrane potential to updated value
        global_parameters->readout_V_1 = result;
    }
//    if (neuron_index == 0){
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = global_parameters->readout_V_0;
//    }
//    else if (neuron_index == 1){
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = global_parameters->readout_V_1;
//    }
//    io_printf(IO_BUF, "state = %u - %u\n", current_state, time);
    if (cue_number == 0 && completed_broadcast){ // reset start of new test
        io_printf(IO_BUF, "time entering reset %u\n", time);
//        io_printf(IO_BUF, "Resetting\n");
        completed_broadcast = false;
        current_time = time;
        current_state = STATE_CUE;
        accumulative_direction = 0;
        // error params
        global_parameters->cross_entropy = 0.k;
        learning_signal = 0.k;
        global_parameters->mean_0 = 0.k;
        global_parameters->mean_1 = 0.k;
        softmax_0 = 0k;
        softmax_1 = 0k;
        while (!spin1_send_mc_packet(
                key | neuron_index,  bitsk(global_parameters->cross_entropy), 1 )) {
            spin1_delay_us(1);
        }
    }
//    io_printf(IO_BUF, "current_state = %u, cue_number = %u, direction = %u, time = %u\n", current_state, cue_number, current_cue_direction, time);
    // In this state the environment is giving the left/right cues to the agent
    if (current_state == STATE_CUE){
//        io_printf(IO_BUF, "time entering cue %u\n", time);
        if (neuron_index == 0){
            // if it's current in the waiting time between cues do nothing
//            if ((time - current_time) % (wait_between_cues + duration_of_cue) < wait_between_cues){
//                 do nothing?
//            }
            // begin sending left/right cue
            if ((time - current_time) % (wait_between_cues + duration_of_cue) >= wait_between_cues){
                // pick broadcast if just entered
                if ((time - current_time) % (wait_between_cues + duration_of_cue) == wait_between_cues){
                    // pick new value and broadcast
//                    REAL random_value = (REAL)(mars_kiss64_seed(global_parameters->kiss_seed) / (REAL)0xffffffff); // 0-1
//                    if (random_value < 0.5k){
//                        current_cue_direction = 0;
//                    }
//                    else{
//                        current_cue_direction = 1;
//                    }
                    current_cue_direction = (current_cue_direction + 1) % 2;
                    accumulative_direction += current_cue_direction;
                    REAL payload;
                    payload = global_parameters->rate_on;
//                    io_printf(IO_BUF, "poisson setting 1, direction = %u\n", current_cue_direction);
                    for (int j = current_cue_direction*global_parameters->p_pop_size;
                            j < current_cue_direction*global_parameters->p_pop_size + global_parameters->p_pop_size; j++){
                        spin1_send_mc_packet(global_parameters->p_key | j, bitsk(payload), WITH_PAYLOAD);
                    }
                }
            }
            // turn off and reset if finished
            else if ((time - current_time) % (wait_between_cues + duration_of_cue) == 0 && (time - current_time) > 0){//(wait_between_cues + duration_of_cue) - 1){
                cue_number += 1;
                REAL payload;
                payload = global_parameters->rate_off;
//                    io_printf(IO_BUF, "poisson setting 2, direction = %u\n", current_cue_direction);
                for (int j = current_cue_direction*global_parameters->p_pop_size;
                        j < current_cue_direction*global_parameters->p_pop_size + global_parameters->p_pop_size; j++){
                    spin1_send_mc_packet(global_parameters->p_key | j, bitsk(payload), WITH_PAYLOAD);
                }
                if (cue_number >= total_cues){
                    current_state = (current_state + 1) % 3;
                }
            }
        }
    }
    else if (current_state == STATE_WAITING){
//        io_printf(IO_BUF, "time entering wait %u\n", time);
        // waiting for prompt, all things ok
        if (cue_number >= total_cues){
            current_time = time;
            cue_number = 0;
        }
        if ((time - current_time) >= wait_before_result){
            current_state = (current_state + 1) % 3;
            start_prompt = true;
        }
    }
    else if (current_state == STATE_PROMPT){
//        io_printf(IO_BUF, "time entering prompt %u\n", time);
        if (start_prompt && neuron_index == 1){
            current_time = time;
            // send packets to the variable poissons with the updated states
            for (int i = 0; i < 4; i++){
                REAL payload;
                payload = global_parameters->rate_on;
//                io_printf(IO_BUF, "poisson setting 3, turning on prompt\n");
                for (int j = 2*global_parameters->p_pop_size;
                        j < 2*global_parameters->p_pop_size + global_parameters->p_pop_size; j++){
                    spin1_send_mc_packet(global_parameters->p_key | j, bitsk(payload), WITH_PAYLOAD);
                }
            }
        }
        if (neuron_index == 2){ // this is the error source
            // Switched to always broadcasting error but with packet
//            ticks_for_mean += 1; //todo is it a running error like this over prompt?
            start_prompt = false;
//            io_printf(IO_BUF, "maybe here - %k - %k\n", global_parameters->mean_0, global_parameters->mean_1);
//            io_printf(IO_BUF, "ticks %u - accum %k - ", ticks_for_mean, (accum)ticks_for_mean);
            // Softmax of the exc and inh inputs representing 1 and 0 respectively
            // may need to scale to stop huge numbers going in the exp
//            io_printf(IO_BUF, "v0 %k - v1 %k\n", global_parameters->readout_V_0, global_parameters->readout_V_1);
//            global_parameters->mean_0 += global_parameters->readout_V_0;
//            global_parameters->mean_1 += global_parameters->readout_V_1;
            // divide -> * 1/x
//            io_printf(IO_BUF, " umm ");
//            accum exp_0 = expk(global_parameters->mean_0 / (accum)ticks_for_mean);
//            accum exp_1 = expk(global_parameters->mean_1 / (accum)ticks_for_mean);
            accum exp_0 = expk(global_parameters->readout_V_0 * 0.1k);
            accum exp_1 = expk(global_parameters->readout_V_1 * 0.1k);
//            io_printf(IO_BUF, "or here - ");
            if (exp_0 == 0k && exp_1 == 0k){
                if (global_parameters->readout_V_0 > global_parameters->readout_V_1){
                    softmax_0 = 10k;
                    softmax_1 = 0k;
                }
                else{
                    softmax_0 = 0k;
                    softmax_1 = 10k;
                }
            }
            else{
//                accum denominator = 1.k  / (exp_1 + exp_0);
                softmax_0 = exp_0 / (exp_1 + exp_0);
                softmax_1 = exp_1 / (exp_1 + exp_0);
            }
//            io_printf(IO_BUF, "soft0 %k - soft1 %k - v0 %k - v1 %k\n", softmax_0, softmax_1, global_parameters->readout_V_0, global_parameters->readout_V_1);
            // What to do if log(0)?
            if (accumulative_direction > total_cues >> 1){
                global_parameters->cross_entropy = -logk(softmax_1);
                learning_signal = softmax_0;
                is_it_right = 1;
            }
            else{
                global_parameters->cross_entropy = -logk(softmax_0);
                learning_signal = softmax_0 - 1.k;
                is_it_right = 0;
            }
//            if (softmax_0 > 0.5){
//                choice = 0;
//            }
//            else{
//                choice = 1;
//            }
            while (!spin1_send_mc_packet(
                    key | neuron_index,  bitsk(learning_signal), 1 )) {
                spin1_delay_us(1);
            }
//            if(learning_signal){
//                io_printf(IO_BUF, "learning signal before cast = %k\n", learning_signal);
//            }
//            learning_signal = global_parameters->cross_entropy;
//            recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] =
//            io_printf(IO_BUF, "broadcasting error\n");
        }
        if ((time - current_time) >= prompt_duration && neuron_index == 0){
//            io_printf(IO_BUF, "time entering end of test %u\n", time);
//            io_printf(IO_BUF, "poisson setting 4, turning off prompt\n");
            current_state = 0;
            completed_broadcast = true;
            for (int i = 0; i < 4; i++){
                REAL payload;
                payload = global_parameters->rate_off;
                for (int j = 2*global_parameters->p_pop_size;
                        j < 2*global_parameters->p_pop_size + global_parameters->p_pop_size; j++){
                    spin1_send_mc_packet(global_parameters->p_key | j, payload, WITH_PAYLOAD);
                }
            }
        }
    }

//    learning_signal = global_parameters->cross_entropy;

    recorded_variable_values[GSYN_INHIBITORY_RECORDING_INDEX] = learning_signal;
    recorded_variable_values[V_RECORDING_INDEX] = voltage;
//    recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = ;
//    if (neuron_index == 2){
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = accumulative_direction;
//    }
//    else {
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = 3.5;
//    }
    if (neuron_index == 2){ //this neuron does nothing
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = neuron->syn_state[90].z_bar;
//        recorded_variable_values[V_RECORDING_INDEX] = neuron->syn_state[90].z_bar;
        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = neuron->syn_state[90].delta_w;
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = is_it_right;
    }
    else if (neuron_index == 1){
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = neuron->syn_state[55].z_bar;
//        recorded_variable_values[V_RECORDING_INDEX] = neuron->syn_state[55].z_bar;
        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = neuron->syn_state[40].delta_w;
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = softmax_0;
    }
    else{
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = neuron->syn_state[1].z_bar;
//        recorded_variable_values[V_RECORDING_INDEX] = neuron->syn_state[1].z_bar;
        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = neuron->syn_state[0].delta_w;
//        recorded_variable_values[GSYN_EXCITATORY_RECORDING_INDEX] = softmax_0;
    }

    // If spike occurs, communicate to relevant parts of model
    if (spike) {
        // Call relevant model-based functions
        // Tell the neuron model
//        neuron_model_has_spiked(neuron);

        // Tell the additional input
        additional_input_has_spiked(additional_input);
    }

    // Shape the existing input according to the included rule
    synapse_types_shape_input(synapse_type);

    #if LOG_LEVEL >= LOG_DEBUG
        neuron_model_print_state_variables(neuron);
    #endif // LOG_LEVEL >= LOG_DEBUG

    // Return the boolean to the model timestep update
    return spike;
}





//! \brief stores neuron parameter back into sdram
//! \param[in] address: the address in sdram to start the store
static void neuron_impl_store_neuron_parameters(
        address_t address, uint32_t next, uint32_t n_neurons) {
    log_debug("writing parameters");

    //log_debug("writing neuron global parameters");
    spin1_memcpy(&address[next], global_parameters,
            sizeof(global_neuron_params_t));
    next += (sizeof(global_neuron_params_t) + 3) / 4;

    log_debug("writing neuron local parameters");
    spin1_memcpy(&address[next], neuron_array,
            n_neurons * sizeof(neuron_t));
    next += ((n_neurons * sizeof(neuron_t)) + 3) / 4;

    log_debug("writing input type parameters");
    spin1_memcpy(&address[next], input_type_array,
            n_neurons * sizeof(input_type_t));
    next += ((n_neurons * sizeof(input_type_t)) + 3) / 4;

    log_debug("writing threshold type parameters");
    spin1_memcpy(&address[next], threshold_type_array,
            n_neurons * sizeof(threshold_type_t));
    next += ((n_neurons * sizeof(threshold_type_t)) + 3) / 4;

    log_debug("writing synapse parameters");
    spin1_memcpy(&address[next], neuron_synapse_shaping_params,
            n_neurons * sizeof(synapse_param_t));
    next += ((n_neurons * sizeof(synapse_param_t)) + 3) / 4;

    log_debug("writing additional input type parameters");
    spin1_memcpy(&address[next], additional_input_array,
            n_neurons * sizeof(additional_input_t));
    next += ((n_neurons * sizeof(additional_input_t)) + 3) / 4;
}

#if LOG_LEVEL >= LOG_DEBUG
void neuron_impl_print_inputs(uint32_t n_neurons) {
	bool empty = true;
	for (index_t i = 0; i < n_neurons; i++) {
		empty = empty
				&& (bitsk(synapse_types_get_excitatory_input(
						&(neuron_synapse_shaping_params[i]))
					- synapse_types_get_inhibitory_input(
						&(neuron_synapse_shaping_params[i]))) == 0);
	}

	if (!empty) {
		log_debug("-------------------------------------\n");

		for (index_t i = 0; i < n_neurons; i++) {
			input_t input =
				synapse_types_get_excitatory_input(
					&(neuron_synapse_shaping_params[i]))
				- synapse_types_get_inhibitory_input(
					&(neuron_synapse_shaping_params[i]));
			if (bitsk(input) != 0) {
				log_debug("%3u: %12.6k (= ", i, input);
				synapse_types_print_input(
					&(neuron_synapse_shaping_params[i]));
				log_debug(")\n");
			}
		}
		log_debug("-------------------------------------\n");
	}
}

void neuron_impl_print_synapse_parameters(uint32_t n_neurons) {
	log_debug("-------------------------------------\n");
	for (index_t n = 0; n < n_neurons; n++) {
	    synapse_types_print_parameters(&(neuron_synapse_shaping_params[n]));
	}
	log_debug("-------------------------------------\n");
}

const char *neuron_impl_get_synapse_type_char(uint32_t synapse_type) {
	return synapse_types_get_type_char(synapse_type);
}
#endif // LOG_LEVEL >= LOG_DEBUG

#endif // _NEURON_IMPL_LEFT_RIGHT_READOUT_H_
