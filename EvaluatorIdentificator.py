#The object EvaluatorIdentificator contains all the information to identify the evaluator, and this information is
# recorded in the results. Such objects are used to record the specific experimental design used during a test
# (the number of simulation scenarios, the time horizon, and the policy used to update decisions when new information
# is available.).
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
