Model
=====
* generate spikes

Neurons
=======

* neurons
  * position
  * type (inhibitory: PV, VIP, SOM; excitatory)
* synapses
  * weight

* model parameters (time constants, resting potential)

Simulation
============


Measurement
===========

* electrodes
  * position 

for the cortex spikes:

inputs from cortex
------------------
cortex spikes (which neurons fire) * cortex connectivity -> activations (voltage, v_exc) -> currents

voltage = (cortex current + vip currents + som currents + ...)

