# env_handler.py
#
# Designed to help with setting up and utilizing an .ini file so that several people can easily run
#   the suite of tools in this repository.

from configparser import ConfigParser
from pathlib import Path

CONFIG_FILE = "env.ini"

def write_config():
    new_config = ConfigParser()
    new_config['bmi_analysis'] = {
        'patient_data': '',
    }

    with open(CONFIG_FILE, 'w') as fp:
        new_config.write(fp)

def bad_config():
    print(f"Bad configuration file detected, writing {CONFIG_FILE} in cwd with required parameters.")
    write_config()
    exit(-1)

def get_patient_dir():
    config = ConfigParser()
    try:
        config.read(CONFIG_FILE)
        return Path(config.get('bmi_analysis', 'patient_data'))
    except:
        bad_config()