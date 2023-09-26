from src.entities.aircraft import aircraft


class vertiport:
    def __init__(self, aircarft_initial_soc, initial_fleet_size, flight_time, location, time_step):
        self.location = location # Location of the vertiport: LAX(0) or DTLA(1)
        self.idle_aircraft = [aircraft(location, aircarft_initial_soc, flight_time, time_step) for i in range(initial_fleet_size)]  # List to store Aircraft instances
        self.charging_aircraft = []
        self.in_flight_aircraft = []

    def collect_arriving_aircraft(self, origin_vertiport):
        if self.location == 0:
            for idx, aircraft in enumerate(origin_vertiport.in_flight_aircraft):
                if aircraft.location == 0:
                    aircraft.direction = None
                    self.idle_aircraft.append(aircraft)
                    origin_vertiport.in_flight_aircraft.pop(idx)

        if self.location == 1:
            for idx, aircraft in enumerate(origin_vertiport.in_flight_aircraft):
                if aircraft.location == 1:
                    aircraft.direction = None
                    self.idle_aircraft.append(aircraft)
                    origin_vertiport.in_flight_aircraft.pop(idx)

    def collect_charging_aircraft(self):
        self.idle_aircraft = self.idle_aircraft + self.charging_aircraft
        self.charging_aircraft = []

    
    def sort_idle_aircraft(self):
        self.idle_aircraft = sorted(self.idle_aircraft, key=lambda aircraft: aircraft.soc, reverse=True)


    def dispatch_aircraft_for_flight(self, num_aircraft_dispatch):
        for idx, aircraft in enumerate(self.idle_aircraft):
            if self.location == 0:
                aircraft.direction = 1
            elif self.location == 1:
                aircraft.direction = 0

            self.in_flight_aircraft.append(aircraft)
            self.idle_aircraft.pop(idx)

            if idx == (num_aircraft_dispatch-1):
                break
        
    
    def commit_aircraft_to_charging(self, num_aircraft_charging):
        for idx, aircraft in enumerate(self.idle_aircraft):
            aircraft.direction = 'charging'
            self.charging_aircraft.append(aircraft)
            self.idle_aircraft.pop(idx)

            if idx == (num_aircraft_charging-1):
                break