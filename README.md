# Droneport Traffic Control

Droneport is a system for autonomous drone battery management. It consists of both hardware and software components. One of the software components is Droneport Traffic Control (DPTC), which takes care of controlling the whole system, i.e., it sends commands to both drones and ports. Droneport Commander, the subsystem of DPTC, is responsible for communication with drones and Droneport devices. It also includes simple user interface that can be operated via a terminal. Another part of the DPTC is the so-called Droneport Orchestrator, whose purpose is to schedule the drone's battery replacement with respect to the specified missions and battery status. 

The component is in the early experimental phase. Droneport Orchestrator is based on the work described in 
>Song, B. D., Kim, J., & Morrison, J. R. (2016). Rolling Horizon Path Planning of an Autonomous System of UAVs for Persistent Cooperative Service: MILP Formulation and Efficient Heuristics. Journal of Intelligent & Robotic Systems, 84(1–4), 241–258. https://doi.org/10.1007/s10846-015-0280-5.

In orchestrator.ipynb a MILP version with the original parameters is implemented, in orchestrator-lite.ipynb a simpler task with two drones and two missions will be implemented, where each drone has its own independent mission.

In trajectory_planner.ipynb a nonlinear time-continous optimal control problem of trajectory planning considering the battery state of charge is solved.  The problem is transcribed using chebyshev pseudospectral collocation method to nonlinear program (NLP), which is solved with IPOPT solver.