#This object contains all the information which allow to identify the test
from __future__ import absolute_import, division, print_function

class TestIdentificator( object ):

    # Constructor
    def __init__(self, instancename, model, method, sampling, nrscenario, scenarioseed, useevpi, mipsetting):
        self.InstanceName = instancename
        self.Model = model
        self.Method = method
        self.ScenarioSampling = sampling
        self.NrScenario = nrscenario
        self.ScenarioSeed = scenarioseed
        self.EVPI = useevpi
        self.MIPSetting = mipsetting

    def GetAsStringList(self):
        result = [self.InstanceName,
                  self.Model,
                  self.Method,
                  self.ScenarioSampling,
                  self.NrScenario,
                  "%s"%self.ScenarioSeed,
                  "%s"%self.EVPI,
                  self.MIPSetting]
        return result

    def GetAsString(self):
        result = "_".join( [self.InstanceName,
                            self.Model,
                            self.Method,
                            self.ScenarioSampling,
                            self.NrScenario,
                            "%s"%self.ScenarioSeed,
                            "%s"%self.EVPI,
                            self.MIPSetting])
        return result
