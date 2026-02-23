import numpy as np

def compute_specific_humidity_from_relative_humidity(temperature, \
                                        relative_humidity):
    """Compute relative humidity from temperature and specific humidity

    Args:
        temperature (numpy array): temperature in degree celcius
        relative_humidity (numpy array): relative humidity in percentage

    Returns:
        numpy array: relative humidity in g/kg
    """

    vapor_pressure = 610.94 * np.exp(temperature * 17.625 / \
                                     (temperature + 243.04))
    ws = vapor_pressure * 0.622 / (101325.0 - vapor_pressure)

    w = relative_humidity * ws / 100

    specific_humidity = w / (1 + w) 

    return 1000 * specific_humidity