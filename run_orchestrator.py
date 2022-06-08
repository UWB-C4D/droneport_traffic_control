from orchestrator_object import Battery, Drone, Droneport
from matplotlib import pyplot as plt
from astar import *

# Generate drones and ports

# params for generator
n_drones = 6  # no. of drones
samples = 50  # no. of mission waypoints
size_map = 20  # size of square map
min_speed = 4  # minimum speed of generated drone
var_speed = 2  # variable value of speed
min_ftime = 15*60  # minimum flight time
var_ftime = 5*60  # variable flight time
min_cap = 0.8  # minimum battery capacity
var_cap = 0.2  # variable battery capacity
min_r = 3  # minimum radius of elliptical mission
var_r = 3  # minimum radius of elliptical mission

battery_id = 300

# get random drones
drones = []
for i in range(n_drones):
    battery_id = battery_id + 1
    drones.append(Drone(target_system=i+1, battery=Battery(id=battery_id, current_cap=min_cap+var_cap*np.random.random()),
                        location=size_map*(np.random.random((1, 2))-0.5),
                        max_speed=min_speed+var_speed*np.random.random(),
                        max_flight_time=min_ftime+var_ftime*np.random.random()))
    # init mission and time plan
    drones[-1].gen_circ_rand_mission(samples, min_r, var_r)

n_ports = 5  # no. of Droneport platforms
n_slots = 10  # no. of battery charging slots in Droneport platforms
port_loc_scale = 0.5  # scale for position of platform
locations = port_loc_scale*size_map * \
    np.array([[1, -1], [-1, -1], [-1, 1], [1, 1], [0, 0]]
             )  # x-y coordinatex of dronports


# get random Droneport platforms
ports = []
for i in range(n_ports):
    batteries = []
    # get fully loaded batteries
    for b in range(n_slots):
        battery_id = battery_id + 1
        batteries.append(Battery(id=battery_id))
    ports.append(Droneport(target_system=i+201, batteries=batteries,
                           location=locations[i, :]))


for drone in drones:
    plt.plot(drone.location[0, 0], drone.location[0, 1], 'x')
    plt.plot(drone.mission_items[:, 0], drone.mission_items[:, 1])
plt.plot(locations[:, 0], locations[:, 1], 'r+')
plt.axis('square')
plt.xlabel('x[m]')
plt.ylabel('y[m]')
plt.title('Map with drone mission and Droneport locations')


# Find solution with A*

# no mission progress
start = np.zeros(n_drones, dtype=int)
# every mission is completed
end = np.zeros(n_drones, dtype=int)
for drone in drones:
    end[drone.id-1] = drone.mission_count-1

# every mission SEQ:
n_seq = 0
max_seq = 0
for drone in drones:
    n_seq = n_seq + drone.mission_count
    if drone.mission_count > max_seq:
        max_seq = drone.mission_count
# what squares do we search
actions = -1*np.ones((n_ports*n_seq, 3), dtype=int)
actions_cost = -1*np.ones((n_drones, n_ports, max_seq))
actions_time = -1*np.ones((n_drones, n_ports, max_seq))
k = 0
for drone in drones:
    for port in ports:
        soc, time = drone.soc_and_time2point(port.location)
        for i in range(drone.mission_count):
            actions[k, :] = [drone.id, port.id, i]
            actions_cost[drone.id-1, port.id-201, i] = soc[i]
            actions_time[drone.id-1, port.id-201, i] = time[i]
            k = k+1
# find solution
path = astar(drones, ports, start, end, actions, actions_cost, actions_time)

if path is not None:
    print("plan:")
    for point in path:
        print(point.action)
    
    plt.figure()
    plt.plot(path[-1].missions_time[1]/60.0, path[-1].soc_progress[1]*100)
    plt.ylim(20,100)
    plt.xlabel('t[min]')
    plt.ylabel('SoC[%]')
    plt.title('State of Charge prediction')


if path is not None:
    print("schedule:")
    for window in path[-1].schedule:
        print(window)
        
    print("mission time:")
    for drone in drones:
        print(f"{path[-1].missions_time[drone.id-1][-1,0]/60}min")
        
    
        
plt.show()