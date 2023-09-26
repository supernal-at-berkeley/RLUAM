def logger(Env):
    print('SOC of aircraft in flight from 0-1')
    soc_01 = []
    for aircraft in Env.vertiports[0].in_flight_aircraft:
        soc_01.append(aircraft.soc)
    print(soc_01)

    soc_10 = []
    print('SOC of aircraft in flight from 1-0')
    for aircraft in Env.vertiports[1].in_flight_aircraft:
        soc_10.append(aircraft.soc)
    print(soc_10)