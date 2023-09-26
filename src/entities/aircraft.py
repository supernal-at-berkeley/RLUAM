class AircraftPositionError(Exception):
    pass

class aircraft:
    def __init__(self, location, aircraft_initial_soc, flight_time, time_step):
        self.soc = aircraft_initial_soc
        self.flight_time = flight_time
        self.time_step = time_step

        self.location = location
        self.direction = None

    def fly(self):
        if self.direction == 1:
            self.location += self.time_step/self.flight_time[0,1]
            self.soc -= 0.01
        if self.direction == 0:
            self.location -= self.time_step/self.flight_time[1,0]
            self.soc -= 0.01

    def charge(self, S=-350/80, M=350):
        """
        Aircraft Charging Model

        Args:
            delta_t (int): Charging time in minutes. This should match the time step of the environment
            S (float): Slope of the linear approximation of the line of charging power vs. soc
            M (int): Maximum charging power

        Returns:
            Update the aircraft soc
        """
        if (self.location == 1) | (self.location == 0):
            soc_gained = (M+S*(self.soc*100-20)) * (self.time_step/60)
            self.soc += soc_gained
            self.direction = None
        else:
            raise AircraftPositionError("Must charge airports at vertiports")
        
    
        