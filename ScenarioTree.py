from __future__ import absolute_import, division, print_function

from ScenarioTreeNode import ScenarioTreeNode
from Scenario import Scenario
import cPickle as pickle
import numpy as np
from RQMCGenerator import RQMCGenerator
from Constants import Constants
import math
#from matplotlib import pyplot as PLT

class ScenarioTree(object):
    #Constructor
    def __init__(self, instance=None, branchperlevel=[], seed=-1, mipsolver=None, evaluationscenario = False, averagescenariotree = False,  givenfirstperiod = [], scenariogenerationmethod="MC", generateasYQfix = False, model = "YFix", CopyscenariofromYFIX=False, issymetric = False, givenscenarioset = [] ):
        self.CopyscenariofromYFIX = CopyscenariofromYFIX
        self.Seed = seed
        if Constants.Debug:
            print("Create a tree with seed %r structure: %r"%(seed, branchperlevel))
        np.random.seed(seed)
        self.Nodes = []
        self.Owner = mipsolver
        self.Instance = instance
        self.TreeStructure = branchperlevel
        self.NrBranches = branchperlevel
        self.EvaluationScenrio = evaluationscenario
        self.AverageScenarioTree = averagescenariotree
        self.ScenarioGenerationMethod = scenariogenerationmethod

        #For some types of evaluation, the demand of the  first periods are given and the rest is stochastic
        self.GivenFirstPeriod = givenfirstperiod
        self.FollowGivenUntil = len(self.GivenFirstPeriod)

        #In case the scenario tree has to be the same aas the two stage (YQFix) scenario tree.
        self.GenerateasYQfix = generateasYQfix
        self.Distribution = instance.Distribution
        self.DemandToFollow = []

        self.IsSymetric = issymetric

        #Generate the demand of YFix, then replicate them in the generation of the scenario tree
        if self.GenerateasYQfix:
            treestructure = [1,2] + [1] * (instance.NrTimeBucket-1) + [0]
            YQFixTree = ScenarioTree(instance, treestructure, seed, scenariogenerationmethod=self.ScenarioGenerationMethod)
            YQFixSceanrios = YQFixTree.GetAllScenarios(computeindex=False)
            self.DemandToFollow = [[[YQFixSceanrios[w].Demands[t][p]
                                     for p in self.Instance.ProductSet]
                                    for t in self.Instance.TimeBucketSet]
                                   for w in range(len(YQFixSceanrios))]

        self.DemandYQFixRQMC = []
        self.Model = model
        self.GenerateRQMCForYQFix = (Constants.IsQMCMethos(self.ScenarioGenerationMethod) and self.Model == Constants.ModelYQFix)

        firstuknown = len(self.GivenFirstPeriod)
        firststochastic = max(self.Instance.NrTimeBucketWithoutUncertaintyBefore, firstuknown)
        timebucketswithuncertainty = range(firststochastic,
                                           self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)
        nrtimebucketswithuncertainty = len(timebucketswithuncertainty)

        if self.ScenarioGenerationMethod == Constants.All and model == Constants.ModelYQFix:
            self.GenerateDemandToFollowAll(firststochastic, nrtimebucketswithuncertainty)

        if Constants.IsQMCMethos(self.ScenarioGenerationMethod) and self.GenerateRQMCForYQFix:
            self.GenerateDemandToFollowRQMC(firststochastic, firstuknown, timebucketswithuncertainty,nrtimebucketswithuncertainty)

        if self.CopyscenariofromYFIX:
            self.GenerateDemandToFollowFromScenarioSet( givenscenarioset )

        if self.IsSymetric:
            self.SymetricDemand=[ [] for t in self.Instance.TimeBucketSet]
            self.SymetricProba=[ [] for t in self.Instance.TimeBucketSet]
            for t in self.Instance.TimeBucketSet:
                    self.SymetricDemand[t], self.SymetricProba[t] = \
                        ScenarioTreeNode.CreateDemandNormalDistributiondemand(self.Instance, t, self.TreeStructure[t+1],
                                                                              t < self.Instance.NrTimeBucketWithoutUncertaintyBefore,
                                                                              self.ScenarioGenerationMethod)

        ScenarioTreeNode.NrNode = 0
        #Create the tree:
        self.RootNode = ScenarioTreeNode(owner=self,
                                         instance=instance,
                                         mipsolver=self.Owner,
                                         time=-1, nrbranch=1,
                                         proabibilty=1,
                                         averagescenariotree=True)
        if instance is None:
            self.NrLevel = -1
        else:
            self.NrLevel = instance.NrTimeBucket
        self.NrNode = ScenarioTreeNode.NrNode
        self.Renumber()

    # Generate the demand to follow in two stage when these demand are generated with a tree with RQMC scenario
    def GenerateDemandToFollowRQMC(self, firststochastic, firstuknown, timebucketswithuncertainty,nrtimebucketswithuncertainty):
        avgvector = [self.Instance.ForecastedAverageDemand[t][p]
                     for p in self.Instance.ProductWithExternalDemand
                     for t in timebucketswithuncertainty]
        stdvector = [self.Instance.ForcastedStandardDeviation[t][p]
                     for p in self.Instance.ProductWithExternalDemand
                     for t in timebucketswithuncertainty]
        dimension = len(self.Instance.ProductWithExternalDemand) * (nrtimebucketswithuncertainty)

        nrscenario = max(self.NrBranches[i] for i in range(len(self.NrBranches)))
        rqmcpoint01 = RQMCGenerator.RQMC01(nrscenario, dimension, withweight=True,
                                           QMC=(self.ScenarioGenerationMethod == Constants.QMC))
        #print(rqmcpoint01)
        #print(min(rqmcpoint01[a][b] for a in range(nrscenario) for b in range(dimension) ))
        rmcpoint = ScenarioTreeNode.TransformInverse(rqmcpoint01, nrscenario, dimension, self.Instance.Distribution,
                                                     avgvector, stdvector)

        self.DemandYQFixRQMC = [[[rmcpoint[self.Instance.ProductWithExternalDemandIndex[p] * nrtimebucketswithuncertainty \
                                           + (t - firststochastic)][s]
                                  if self.Instance.HasExternalDemand[p] and t >= firststochastic
                                  else 0.0
                                  for p in self.Instance.ProductSet]
                                 for t in self.Instance.TimeBucketSet]
                                for s in range(nrscenario)]
        for s in range(nrscenario):
            for t in range(self.Instance.NrTimeBucketWithoutUncertaintyBefore, firstuknown):
                for p in self.Instance.ProductSet:
                    self.DemandYQFixRQMC[s][t][p] = self.GivenFirstPeriod[t][p]

    def GenerateDemandToFollowFromScenarioSet(self, scenarioset):

        nrscenario = len(scenarioset)

        self.DemandToFollowMultipleSceario = [[[scenarioset[s].Demands[t][p]
                                                if self.Instance.HasExternalDemand[p]
                                                else 0.0
                                                for p in self.Instance.ProductSet]
                                               for t in self.Instance.TimeBucketSet]
                                              for s in range(nrscenario)]
        self.ProbabilityToFollowMultipleSceario = [scenarioset[s].Probability for s in range(nrscenario)]

    # Generate the demand to follow in two stage when these demand are generated with a tree with all scenario
    def GenerateDemandToFollowAll(self, firststochastic, nrtimebucketswithuncertainty):
        sizefixed = max(len(self.GivenFirstPeriod) - self.Instance.NrTimeBucketWithoutUncertaintyBefore, 0)

        nrscenario = int(max(math.pow(8, 4 - sizefixed), 1))
        temporarytreestructur = [1] + [1] * firststochastic + [8] * (nrtimebucketswithuncertainty) + [0]
        if nrscenario == 1:
            temporarytreestructur = [1, 1, 1, 1, 0]

        temporaryscenariotree = ScenarioTree(self.Instance, temporarytreestructur, self.Seed,
                                             averagescenariotree=False,
                                             scenariogenerationmethod=Constants.All,
                                             givenfirstperiod=self.GivenFirstPeriod)
        temporaryscenarios = temporaryscenariotree.GetAllScenarios(False)

        self.GenerateDemandToFollowFromScenarioSet( temporaryscenarios )

    #This function number the node from highest level to lowest.
    def Renumber(self):
        k = 1
        #Traverse the node per level
        nrlevel = max(n.Time for n in self.Nodes)
        for l in range( nrlevel + 1 ):
            #get the set of node in the level
            nodes = [n for n in self.Nodes if n.Time == l]
            for n in nodes:
                n.NodeNumber = k
                k = k + 1

    #Compute the index of the variable (one variable for each node of the tree)
    def ComputeVariableIdicies(self):
        for n in self.Nodes:
            n.ComputeVariableIndex()

    #Print the scenario tree
    def Display(self):
        print("Print the tree: ")
        self.RootNode.Display()

    #This function assemble the data in the tree, and return the list of leaves, which contain the scenarios
    def GetAllScenarios(self, computeindex=True,expandfirststage = False):
        #A mip solver is required to compute the index, it is not always set
        if computeindex:
            self.ComputeVariableIdicies()
        self.RootNode.CreateAllScenarioFromNode()
        #return the set of leaves as they represent the scenario
        scenarioset = [n for n in self.Nodes if len(n.Branches) == 0]

        scenarios = [Scenario(owner=self,
                              demand=s.DemandsInScenario,
                              proabability=s.ProbabilityOfScenario,
                              quantityvariable=s.QuanitityVariableOfScenario,
                              productionvariable=s.ProductionVariableOfScenario,
                              inventoryvariable=s.InventoryVariableOfScenario,
                              backordervariable=s.BackOrderVariableOfScenario,
                              consumptionvariable=s.ConsumptionVariableOfScenario,
                              nodesofscenario=s.NodesOfScenario) for s in scenarioset]
        id = 0
        for s in scenarios:
            s.ScenarioId = id
            id = id + 1

        if expandfirststage:
            for s in scenarios:
                s.ProductionVariable = [[(self.Owner.StartProductionVariable
                                         + s.ScenarioId * self.Instance.NrProduct * len(self.Instance.TimeBucketSet) + self.Instance.NrProduct * (t) + p)
                                        for p in self.Instance.ProductSet]
                                        for t in self.Instance.TimeBucketSet]

                s.ConsumptionVariable = [[[self.Owner.StartConsumptionVariable + \
                                           s.ScenarioId * self.Instance.NrComponentTotal * len(self.Instance.TimeBucketSet) \
                                        +   self.Instance.NrComponentTotal * t \
                                             + sum(self.Instance.NrComponent[k] for k in range(p)) \
                                             + sum(self.Instance.PossibleComponents[p][k] for k in range(q))
                                             if self.Instance.PossibleComponents[p][q]
                                             else -1
                                             for q in self.Instance.ProductSet]
                                            for p in self.Instance.ProductSet]
                                          for t in self.Instance.TimeBucketSet]

                s.QuanitityVariable = [[self.Owner.StartQuantityVariable +\
                                          s.ScenarioId * self.Instance.NrProduct * len(self.Instance.TimeBucketSet)\
                                        + (self.Instance.NrProduct * t + p) \
                                       for p in self.Instance.ProductSet ]
                                       for t in self.Instance.TimeBucketSet]

        return scenarios

    #Save the scenario tree in a file
    def SaveInFile(self, scearionr):
        result = None
        filepath = './Instances/' + self.Instance.InstanceName + '_Scenario%s.pkl'%scearionr
        try:
          with open(filepath, 'wb') as output:
               pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        except:
          print("file %r not found" %(filepath))

    #This function set the quantity to order at each node of the tree as found in the solution given in argument
    def FillQuantityToOrder(self, sol):
        for n in self.Nodes:
            if n.Time >= 0 and n.Time < self.Instance.NrTimeBucket:
                n.QuantityToOrder = sol.get_values([n.QuanitityVariable[p]for p in self.Instance.ProductSet])
                if len(self.Instance.ConsumptionSet) > 0:
                    n.Consumption = sol.get_values([n.ConsumptionVariable[c[1]][c[0]] for c in self.Instance.ConsumptionSet])
                if n.Time > 0:
                    n.InventoryLevel = sol.get_values([n.InventoryVariable[p] for p in self.Instance.ProductSet])
                    n.BackOrderLevel = sol.get_values([n.BackOrderVariable[self.Instance.ProductWithExternalDemandIndex[p]]
                                                       for p in self.Instance.ProductWithExternalDemand])

    #This function set the quantity to order at each node of the tree as found in the solution given in argument
    def FillQuantityToOrderFromMRPSolution(self, sol):
        scenarionr = -1
        for n in self.Nodes:
            if n.Time >= 0 and  n.Time < self.Instance.NrTimeBucket:
                scenarionr = n.OneOfScenario.ScenarioId
                n.QuantityToOrderNextTime = [sol.ProductionQuantity[scenarionr][n.Time][p]
                                            for p in self.Instance.ProductSet]

                n.InventoryLevelNextTime = [sol.InventoryLevel[scenarionr][n.Time][p] if not self.Instance.HasExternalDemand[p] else float('nan')
                                            for p in self.Instance.ProductSet]

                if n.Time >= 1:
                    n.BackOrderLevelTime = [sol.BackOrder[scenarionr][n.Time -1][self.Instance.ProductWithExternalDemandIndex[p]]
                                            for p in self.Instance.ProductWithExternalDemand]

                    n.InventoryLevelTime = [sol.InventoryLevel[scenarionr][n.Time -1][p]
                                             if self.Instance.HasExternalDemand[p] else float('nan')
                                            for p in self.Instance.ProductSet]
