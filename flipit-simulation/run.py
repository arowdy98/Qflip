#!/usr/bin/env python
import sim
import sim_net
import sim2
import yaml
import sys
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

if len(sys.argv) < 2:
    config = 'configs/Q_vs_periodic.yml'
else:
    config = sys.argv[1]

# s = sim.Simulation()
# s = sim_net.Simulation()
s = sim2.Simulation()
configs = yaml.load(open(config),Loader=Loader)
s.run(**configs)