import gym
import pickle
import numpy as np
import random
import pandas as pd
from src.entities.vertiport import vertiport
from utils.autoregressive_pax_arrival_process import autoregressive_possion_rate, pois_generate

flight_time = np.array([[0,10],[10,0]])
initial_fleet_size = np.array([8,8])
aircraft_initial_soc = 1
time_step = 1
pax_arrival_fn = 'data/' + 'full_year_schedule_0926'
pax_waiting_time_beta = 100
charging_beta = 50

class Env(gym.Env):
    def __init__(self, 
                 flight_time=flight_time, 
                 initial_fleet_size=initial_fleet_size,
                 aircraft_initial_soc=aircraft_initial_soc, 
                 time_step=time_step,
                 pax_arrival_fn=pax_arrival_fn,
                 pax_waiting_time_beta=pax_waiting_time_beta,
                 charging_beta=charging_beta):
        
        self.aircraft_initial_soc = aircraft_initial_soc
        self.initial_fleet_size = initial_fleet_size
        self.flight_time = flight_time
        self.time_step = time_step
        self.counter = 0
        self.vertiports = [vertiport(self.aircraft_initial_soc, self.initial_fleet_size[0], self.flight_time, 0, self.time_step),
                           vertiport(self.aircraft_initial_soc, self.initial_fleet_size[1], self.flight_time, 1, self.time_step)]
        self.pax_waiting_time_beta = pax_waiting_time_beta
        self.charging_beta = charging_beta
        self.ob_dim = 36
        self.ac_dim = 4
    

        self.event_time_counter = 0


        with open(pax_arrival_fn, "rb") as fp:
            pax_arrival_process = pickle.load(fp)
        self.lax_dtla_rate, self.dtla_lax_rate = autoregressive_possion_rate(pax_arrival_process)
        self.lax_dtla_arrival, self.dtla_lax_arrival = self.__pax_arrival_realization__(self.lax_dtla_rate, self.dtla_lax_rate)


    def __pax_arrival_realization__(self, lax_dtla_rate, dtla_lax_rate):
        lax_dtla = pois_generate(lax_dtla_rate, alpha=0.75)
        dtla_lax = pois_generate(dtla_lax_rate, alpha=0.75)

        lax_dtla_arrival = []
        for idx, val in enumerate(lax_dtla):
            for _ in range(val):
                lax_dtla_arrival.append(random.randint(0, 59)+idx*60)
        lax_dtla_arrival = np.array(sorted(lax_dtla_arrival))

        dtla_lax_arrival = []
        for idx, val in enumerate(dtla_lax):
            for _ in range(val):
                dtla_lax_arrival.append(random.randint(0, 59)+idx*60)
        dtla_lax_arrival = np.array(sorted(dtla_lax_arrival))

        return lax_dtla_arrival, dtla_lax_arrival

    def __num_pax_arrival__(self, t):
        num_pax_lax_dtla = len(self.lax_dtla_arrival[(self.lax_dtla_arrival <= t) & (self.lax_dtla_arrival > t-1)])
        num_pax_dtla_lax = len(self.dtla_lax_arrival[(self.dtla_lax_arrival <= t) & (self.dtla_lax_arrival > t-1)])
        return num_pax_lax_dtla, num_pax_dtla_lax

    def reset(self):
        self.vertiports = [vertiport(self.aircraft_initial_soc, self.initial_fleet_size[0], self.flight_time, 0, self.time_step),
                           vertiport(self.aircraft_initial_soc, self.initial_fleet_size[1], self.flight_time, 1, self.time_step)]
        self.lax_dtla_arrival, self.lax_dtla_arrival = self.__pax_arrival_realization__(self.lax_dtla_rate, self.dtla_lax_rate)
        self.event_time_counter = 0

    def compute_action(self):
        num_idle_vertiport_0 = len(self.vertiports[0].idle_aircraft)
        num_idle_vertiport_1 = len(self.vertiports[1].idle_aircraft)

        dispatch_at_vertiport_0 = random.randint(0, num_idle_vertiport_0)
        charge_at_vertiport_0 = random.randint(0, num_idle_vertiport_0-dispatch_at_vertiport_0)

        dispatch_at_vertiport_1 = random.randint(0, num_idle_vertiport_1)
        charge_at_vertiport_1 = random.randint(0, num_idle_vertiport_1-dispatch_at_vertiport_1)

        return ((dispatch_at_vertiport_0, charge_at_vertiport_0), (dispatch_at_vertiport_1, charge_at_vertiport_1))
    
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

        num_pax_lax_dtla, num_pax_dtla_lax = self.__num_pax_arrival__(self.event_time_counter)
        self.event_time_counter += 1

        self.vertiports[0].update_pax_in_queue(num_pax_lax_dtla)
        self.vertiports[1].update_pax_in_queue(num_pax_dtla_lax)

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


        lax_vertiport_idling = []
        for aircraft in self.vertiports[0].idle_aircraft:
            lax_vertiport_idling.append(aircraft.soc)
        lax_vertiport_idling = np.array(lax_vertiport_idling)
        lax_vertiport_idling = np.concatenate([lax_vertiport_idling, np.repeat(0, self.initial_fleet_size.sum()-len(lax_vertiport_idling))])

        dtla_vertiport_idling = []
        for aircraft in self.vertiports[1].idle_aircraft:
            dtla_vertiport_idling.append(aircraft.soc)
        dtla_vertiport_idling = np.array(dtla_vertiport_idling)
        dtla_vertiport_idling = np.concatenate([dtla_vertiport_idling, np.repeat(0, self.initial_fleet_size.sum()-len(dtla_vertiport_idling))])

        queue_length = self.vertiports[0].queue + self.vertiports[1].queue
        charging_time = action[0][1] + action[1][1]
        reward = -(self.charging_beta*charging_time+self.pax_waiting_time_beta*queue_length)

        return ({'lax_idle_aircraft':lax_vertiport_idling,
                 'dtla_idle_aircraft':dtla_vertiport_idling,
                 'lax_queue':self.vertiports[0].queue,
                 'dtla_queue':self.vertiports[1].queue},
                 reward)
          

