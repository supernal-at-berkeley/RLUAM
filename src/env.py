import gym
import pickle
import numpy as np
import random
from src.entities.vertiport import vertiport
from utils.autoregressive_pax_arrival_process import autoregressive_possion_rate, pois_generate

flight_time = np.array([[0,10],[10,0]])
initial_fleet_size = np.array([8,8])
aircraft_initial_soc = 1
time_step = 1
pax_arrival_fn = 'data/' + 'full_year_schedule_0926'

class Env(gym.Env):
    def __init__(self, 
                 flight_time=flight_time, 
                 initial_fleet_size=initial_fleet_size,
                 aircraft_initial_soc=aircraft_initial_soc, 
                 time_step=time_step,
                 pax_arrival_fn = pax_arrival_fn):

        self.time_step = time_step
        self.vertiports = [vertiport(aircraft_initial_soc, initial_fleet_size[0], flight_time, 0, time_step),
                           vertiport(aircraft_initial_soc, initial_fleet_size[1], flight_time, 1, time_step)]


        

        with open('data/'+pax_arrival_fn, "rb") as fp:
            pax_arrival_process = pickle.load(fp)
        self.lax_dtla_rate, self.dtla_lax_rate = autoregressive_possion_rate(pax_arrival_process)
        lax_dtla = pois_generate(self.lax_dtla_rate)
        dtla_lax = pois_generate(self.dtla_lax_rate)

        lax_dtla_arrival = []
        for idx, val in enumerate(lax_dtla):
            for _ in range(val):
                lax_dtla_arrival.append(random.randint(0, 59)+idx*60)

        lax_dtla_arrival = sorted(my_list)


    def reset(self):
        pass
        

    def compute_action(self):
        num_idle_vertiport_0 = len(self.vertiports[0].idle_aircraft)
        num_idle_vertiport_1 = len(self.vertiports[1].idle_aircraft)

        dispatch_at_vertiport_0 = random.randint(0, num_idle_vertiport_0)
        charge_at_vertiport_0 = random.randint(0, num_idle_vertiport_0-dispatch_at_vertiport_0)

        dispatch_at_vertiport_1 = random.randint(0, num_idle_vertiport_1)
        charge_at_vertiport_1 = random.randint(0, num_idle_vertiport_1-dispatch_at_vertiport_1)

        return ((dispatch_at_vertiport_0, charge_at_vertiport_0), (dispatch_at_vertiport_1, charge_at_vertiport_1))
    
    def get_pax_arrival(self, t):




    def step(self, action):
        """
        Move Simulation Forward

        Args:
            action: a tuple of 2 tuples, (num_aircraft_dispatch, num_aircraft_charging)

        Returns:
            Update the aircraft soc
        """

        for vertiport_idx, vertiport in enumerate(self.vertiports):
            for aircraft_idx, aircraft in enumerate(vertiport.charging_aircraft):
                aircraft.charge()

            for aircraft_idx, aircraft in enumerate(vertiport.in_flight_aircraft):
                aircraft.fly()

        self.vertiports[0].collect_arriving_aircraft(self.vertiports[1])
        self.vertiports[1].collect_arriving_aircraft(self.vertiports[0])

        self.vertiports[0].collect_charging_aircraft()
        self.vertiports[1].collect_charging_aircraft()

        self.vertiports[0].sort_idle_aircraft()
        self.vertiports[1].sort_idle_aircraft()

        self.vertiports[0].dispatch_aircraft_for_flight(action[0][0])
        self.vertiports[1].dispatch_aircraft_for_flight(action[1][0])

        self.vertiports[0].commit_aircraft_to_charging(action[0][1])
        self.vertiports[1].commit_aircraft_to_charging(action[1][1])


        return None

