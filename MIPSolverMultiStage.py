from __future__ import absolute_import, division, print_function
import MIPSolver

class MIPSolverMultiStage(MIPSolver):

    def __init__(self,
                 instance,
                 model,
                 scenariotree,
                 evpi = False,
                 implicitnonanticipativity = False,
                 givenquantities = [],
                 givensetups=[],
                 fixsolutionuntil = -1,
                 evaluatesolution = False,
                 yfixheuristic = False,
                 demandknownuntil = -1,
                 mipsetting = "",
                 warmstart = False,
                 usesafetystock = False,
                 usesafetystockgrave = False,
                 rollinghorizon = False,
                 logfile = "",
                 givenSGrave = [] ):

        MIPSolver.__init__(self, instance)


    def GetNrQuantityVariable(self):
        return self.Instance.NrProduct * ( self.DemandScenarioTree.NrNode - 1  - self.NrScenario )

    def GetSenarioIndexForQuantity(self, p, t, w):
            return self.Scenarios[w].QuanitityVariable[t][p]

    def GetIndexQuantityVariable(self, p, t, w):
            return self.Scenarios[w].QuanitityVariable[t][p]

