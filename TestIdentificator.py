#This object contains all the information which allow to identify the test
from __future__ import absolute_import, division, print_function

class TestIdentificator( object ):

    # Constructor
    def __init__(self, instancename, model, method, sampling, nrscenario, scenarioseed, useevpi, nrscenarioforward,mipsetting, sddpsetting):
        self.InstanceName = instancename
        self.Model = model
        self.Method = method
        self.ScenarioSampling = sampling
        self.NrScenario = nrscenario
        self.ScenarioSeed = scenarioseed
        self.EVPI = useevpi
        self.MIPSetting = mipsetting
        self.NrScenarioForward = nrscenarioforward
        self.SDDPSetting = sddpsetting


    def GetAsStringList(self):
        result = [self.InstanceName,
                  self.Model,
                  self.Method,
                  self.ScenarioSampling,
                  self.NrScenario,
                  "%s"%self.ScenarioSeed,
                  "%s"%self.EVPI,
                  "%s"%self.NrScenarioForward,
                  self.MIPSetting,
                  self.SDDPSetting]
        return result

    def GetAsString(self):
        result = "_".join([self.InstanceName,
                           self.Model,
                           self.Method,
                           self.ScenarioSampling,
                           self.NrScenario,
                           "%s"%self.ScenarioSeed,
                           "%s"%self.EVPI,
                           "%s"%self.NrScenarioForward,
                           self.MIPSetting,
                           self.SDDPSetting])
        return result
