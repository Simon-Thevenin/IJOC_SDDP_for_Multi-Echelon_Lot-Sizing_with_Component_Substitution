from __future__ import absolute_import, division, print_function
import MIPSolver

class MIPSolverTwoStage(MIPSolver):

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
        return self.Instance.NrProduct * self.Instance.NrTimeBucket

    def GetSenarioIndexForQuantity(self, p, t, w):
        return 0

    def GetIndexQuantityVariable(self, p, t, w):
        return self.GetStartQuantityVariable() + t * self.Instance.NrProduct + p





