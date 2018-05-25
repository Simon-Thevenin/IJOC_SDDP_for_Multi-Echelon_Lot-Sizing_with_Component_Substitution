#This object contains all the information which allow to identify the evaluator
from __future__ import absolute_import, division, print_function

class EvaluatorIdentificator( object ):

    # Constructor
    def __init__( self, policygeneration, nrevaluation, timehorizon, allscenario):
        self.PolicyGeneration = policygeneration
        self.NrEvaluation = nrevaluation
        self.TimeHorizon = timehorizon
        self.AllScenario = allscenario

    def GetAsStringList(self):
        result = [self.PolicyGeneration,
                  "%s" % self.NrEvaluation,
                  "%s" % self.TimeHorizon,
                  "%s" % self.AllScenario]
        return result

    def GetAsString(self):
        result = "_".join(self.GetAsStringList())
        return result
