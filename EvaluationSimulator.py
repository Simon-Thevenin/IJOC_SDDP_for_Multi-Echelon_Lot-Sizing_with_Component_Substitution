#This class provide a framework to evaluate the performance of the method through a simulation
#over a large number of scenarios.

from __future__ import absolute_import, division, print_function
#import pandas as pd
#from matplotlib import pyplot as PLT
from MIPSolver import MIPSolver
from ScenarioTree import ScenarioTree
from Constants import Constants
from DecentralizedMRP import DecentralizedMRP
from ProgressiveHedging import ProgressiveHedging
#from RollingHorizonSolver import RollingHorizonSolver
import time
import math
from datetime import datetime
import csv
from scipy import stats
import numpy as np
import copy
import itertools
#from MRPSolution import MRPSolution
#from decimal import Decimal, ROUND_HALF_DOWN
import pickle
import RollingHorizonSolver
#from matplotlib import pyplot as PLT

class EvaluationSimulator(object):

    #Constructor
    def __init__(self, instance, solutions=[], sddps=[],
                 testidentificator = [],
                 evaluatoridentificator =[],
                 treestructure=[],
                 model="YQFix"):
        self.Instance = instance
        self.Solutions = solutions
        self.SDDPs = sddps
        self.TestIdentificator = testidentificator
        self.EvalatorIdentificator = evaluatoridentificator
        self.NrSolutions = max(len(self.Solutions), len(self.SDDPs))
        self.Policy = evaluatoridentificator.PolicyGeneration
        self.StartSeedResolve = Constants.SeedArray[0]

        self.ScenarioGenerationResolvePolicy = self.TestIdentificator.ScenarioSampling
        self.EVPI = testidentificator.EVPI
        if self.EVPI:
            self.EVPISeed = Constants.SeedArray[0]

        self.MIPResolveTime = [None for t in instance.TimeBucketSet]
        self.IsDefineMIPResolveTime = [False for t in instance.TimeBucketSet]

        self.PHResolveTime = [None for t in instance.TimeBucketSet]
        self.IsDefinePHResolve = [False for t in instance.TimeBucketSet]


        self.ReferenceTreeStructure = treestructure
        self.EvaluateAverage = Constants.IsDeterministic(self.TestIdentificator.Model)
        self.UseSafetyStock = Constants.UseSafetyStock(self.TestIdentificator.Model)
        self.Model = model
        self.UseSafetyStockGrave = (self.TestIdentificator.Model == Constants.AverageSSGrave)

        self.YeuristicYfix = self.TestIdentificator.Model == Constants.ModelHeuristicYFix
        if self.Policy == Constants.RollingHorizon:
            self.RollingHorizonSolver = RollingHorizonSolver(self.Instance,  model , self.ReferenceTreeStructure,
                                                             self.StartSeedResolve, self.ScenarioGenerationResolvePolicy,
                                                             self.EvaluatorIdentificator.TimeHorizon, self.UseSafetyStock, self)

        self.DecentralizedMRP = DecentralizedMRP(self.Instance, Constants.IsRuleWithGrave(self.Model))

    #This function evaluate the performance of a set of solutions obtain with the same method (different solutions due to randomness in the method)
    def EvaluateYQFixSolution(self, saveevaluatetab=False, filename="", evpi=False):

        # Compute the average value of the demand
        nrscenario = self.EvalatorIdentificator.NrEvaluation
        allscenario = self.EvalatorIdentificator.AllScenario
        start_time = time.time()
        Evaluated = [-1 for e in range(nrscenario)]
        Probabilities = [-1 for e in range(nrscenario)]
        OutOfSampleSolution = None
        mipsolver = None
        firstsolution = True
        nrerror = 0

        for n in range(self.NrSolutions):
                sol = None
                if not evpi and not self.Policy == Constants.RollingHorizon:
                   sol = self.Solutions[n]

                if Constants.IsSDDPBased(self.TestIdentificator.Method):
                    sddp = self.SDDPs[n]

                evaluatoinscenarios, scenariotrees = self.GetScenarioSet(Constants.EvaluationScenarioSeed, nrscenario, allscenario)

                if Constants.IsSDDPBased( self.TestIdentificator.Method ) : #== Constants.SDDP:
                     self.ForwardPassOnScenarios(sddp, evaluatoinscenarios, sol)

                firstscenario = True
                self.IsDefineMIPResolveTime = [False for t in self.Instance.TimeBucketSet]
                self.IsDefinePHResolve = [False for t in self.Instance.TimeBucketSet]

                average = 0
                totalproba = 0
                for indexscenario in range(nrscenario):
                    scenario = evaluatoinscenarios[indexscenario]
                    scenariotree = scenariotrees[indexscenario]

                    if not evpi:
                        if self.TestIdentificator.Method == Constants.MIP or self.TestIdentificator.Method == Constants.ProgressiveHedging:
                            givensetup, givenquantty, givenconsumption = self.GetDecisionFromSolutionForScenario(sol, scenario)

                        if Constants.IsSDDPBased(self.TestIdentificator.Method):
                            givensetup, givenquantty, givenconsumption = self.GetDecisionFromSDDPForScenario(sddp, indexscenario)
                            # Solve the MIP and fix the decision to the one given.
                        if Constants.Debug:
                            for t in self.Instance.TimeBucketSet:
                                    print("Setup:%r" % givensetup[t])
                                    print("Quantity:%r" % givenquantty[t])
                                    print("Consumption:%r" % givenconsumption[t])
                                    print("Demand:%r" % scenario.Demands[t])


                    else:
                        givensetup = []
                        givenquantty = []
                        givenconsumption = []

                    if firstscenario:
                        #Defin the MIP
                        if not evpi:
                            mipsolver = MIPSolver(self.Instance, Constants.ModelYQFix, scenariotree,
                                                  evpi=False,
                                                  implicitnonanticipativity=False,
                                                  evaluatesolution=True,
                                                  givenquantities=givenquantty,
                                                  givensetups=givensetup,
                                                  givenconsumption=givenconsumption,
                                                  fixsolutionuntil=self.Instance.NrTimeBucket)
                        else:
                            mipsolver = MIPSolver(self.Instance, self.Model, scenariotree,
                                                  evpi=True)
                        mipsolver.BuildModel()
                    else:
                        #update the MIP
                        mipsolver.ModifyMipForScenarioTree(scenariotree)
                        if not self.Policy == Constants.Fix and not evpi:
                            mipsolver.ModifyMipForFixQuantity(givenquantty)
                            mipsolver.ModifyMipForFixConsumption(givenconsumption)

                        if self.Policy == Constants.RollingHorizon:
                            mipsolver.ModifyMIPForSetup(givensetup)

                    mipsolver.Cplex.parameters.advance = 0
                    mipsolver.Cplex.parameters.simplex.tolerances.feasibility= 0.01
                    #mipsolver.Cplex.parameters.lpmethod = 2
                    mipsolver.Cplex.parameters.lpmethod.set(mipsolver.Cplex.parameters.lpmethod.values.barrier)
                    solution = mipsolver.Solve()

                    #CPLEX should always find a solution due to complete recourse
                    if solution == None:
                        if Constants.Debug:
                            mipsolver.Cplex.write("mrp.lp")
                            raise NameError("error at seed %d with given qty %r"%(indexscenario, givenquantty))
                            nrerror = nrerror + 1
                    else:
                        Evaluated[indexscenario] = solution.TotalCost
                        if allscenario == 0:
                            scenario.Probability = 1.0 / float(nrscenario)
                        Probabilities[indexscenario] = scenario.Probability
                        average += solution.TotalCost * scenario.Probability
                        totalproba += scenario.Probability
                        #Record the obtain solution in an MRPsolution  OutOfSampleSolution
                        if firstsolution:
                            if firstscenario:
                                OutOfSampleSolution = solution
                            else:
                                OutOfSampleSolution.Merge(solution)

                        firstscenario = False

                    if firstsolution:
                        for s in OutOfSampleSolution.Scenarioset:
                            s.Probability = 1.0 / len(OutOfSampleSolution.Scenarioset)


        OutOfSampleSolution.ComputeStatistics()

        duration = time.time() - start_time
        if Constants.Debug:
            print("Duration od evaluation: %r, outofsampl cost:%r total proba:%r" % (duration, average, totalproba)) # %r"%( duration, Evaluated )
        self.EvaluationDuration = duration

        KPIStat = OutOfSampleSolution.PrintStatistics(self.TestIdentificator, "OutOfSample", indexscenario, nrscenario, duration, False, self.Policy )

        #Save the evaluation result in a file (This is used when the evaluation is parallelized)
        if saveevaluatetab:
                with open(filename+"Evaluator.txt", "wb") as fp:
                    pickle.dump(Evaluated, fp)

                with open(filename + "Probabilities.txt", "wb") as fp:
                    pickle.dump(Probabilities, fp)

                with open(filename+"KPIStat.txt", "wb") as fp:
                    pickle.dump(KPIStat, fp)

        if Constants.PrintDetailsExcelFiles:
            namea = self.TestIdentificator.GetAsString()
            nameb = self.EvalatorIdentificator.GetAsString()
            OutOfSampleSolution.PrintToExcel(namea+nameb+".xlsx")


    #This function return the setup decision and quantity to produce for the scenario given in argument
    def GetDecisionFromSolutionForScenario(self, sol,  scenario):

        givenquantty = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        givensetup = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        givenconsumption = [[0 for c in self.Instance.ConsumptionSet] for t in self.Instance.TimeBucketSet]
        if self.Policy == Constants.RollingHorizon:
            givensetup, givenquantty, givenconsumption = self.RollingHorizonSolver.ApplyRollingHorizonSimulation( scenario )

        else:
            # The setups are fixed in the first stage
            givensetup = [[ (sol.Production[0][t][p] ) for p in self.Instance.ProductSet]
                            for t in self.Instance.TimeBucketSet]

            # For model YQFix, the quatities are fixed, and can be taken from the solution
            if self.Policy == Constants.Fix:
                givenquantty = [[sol.ProductionQuantity[0][t][p]
                                 for p in self.Instance.ProductSet]
                                 for t in self.Instance.TimeBucketSet]
                givenconsumption = [[sol.Consumption[0][t][c[0]][c[1]]
                                         for c in self.Instance.ConsumptionSet]
                                         for t in self.Instance.TimeBucketSet]

            # For model YFix, the quantities depend on the scenarion
            else:
                givenquantty = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
                givenconsumption = [[0 for c in self.Instance.ConsumptionSet] for t in self.Instance.TimeBucketSet]


                previousnode = sol.ScenarioTree.RootNode
                #At each time period the quantity to produce is decided based on the demand known up to now


                for ti in self.Instance.TimeBucketSet:
                    demanduptotimet = [[scenario.Demands[t][p] for p in self.Instance.ProductSet] for t in range(ti)]

                    if self.Policy == Constants.Resolve:
                            givenquantty[ti], givenconsumption[ti], error = self.GetQuantityByResolve(demanduptotimet, ti, givenquantty, givenconsumption, sol,  givensetup)

        return givensetup, givenquantty, givenconsumption

    #This method run a forward pass of the SDDP algorithm on the considered set of scenarios
    def ForwardPassOnScenarios(self, sddp, scenarios, solution):
        sddp.EvaluationMode = True
        Constants.SDDPRunSigleTree = False
        sddp.HeuristicSetupValue = [[ solution.Production[0][t][p] for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        # Make a forward pass on the
        #Create the SAA scenario, which are used to compute the EVPI scenario
        sddp.GenerateSAAScenarios2()

        # Get the set of scenarios
        sddp.CurrentSetOfTrialScenarios = scenarios
        sddp.ScenarioNrSet = len(scenarios)
        sddp.CurrentNrScenario = len(scenarios)
        sddp.TrialScenarioNrSet = range(len(sddp.CurrentSetOfTrialScenarios))

        sddp.GenerateStrongCut = False
        # Modify the number of scenario at each stage
        for stage in sddp.StagesSet:
            sddp.ForwardStage[stage].SetNrTrialScenario(len(scenarios))
            sddp.ForwardStage[stage].FixedScenarioPobability = [1]
            sddp.ForwardStage[stage].FixedScenarioSet = [0]
            sddp.BackwardStage[stage].SAAStageCostPerScenarioWithoutCostoGopertrial = [0 for w in
                                                                                       sddp.TrialScenarioNrSet]

        sddp.ForwardStage[0].CopyDecisionOfScenario0ToAllScenario()
        sddp.ForwardPass(ignorefirststage=False)

        sddp.ForwardStage[0].CopyDecisionOfScenario0ToAllScenario()
        if Constants.Debug:
            sddp.ComputeCost()
            sddp.UpdateUpperBound()
            print("Run forward pass on all evaluation scenarios, cost: %r" %sddp.CurrentExpvalueUpperBound)

    # This function return the setup decision and quantity to produce for the scenario given in argument
    def GetDecisionFromPHForScenario(self, demanduptotimet, time, givenquantity, givenconsumption, givensetup):
        if not self.IsDefinePHResolve[time]:
            #Create the set of sub-instances.
            scenariotree, treestructure = self.GetScenarioTreeForResolve(time, demanduptotimet)
            self.PHResolveTime[time] = ProgressiveHedging(self.Instance, self.TestIdentificator, treestructure,
                                                          scenariotree, givensetup=givensetup, fixuntil=time-1)

            for w in [0]:#self.PHResolveTime[time].ScenarioNrSet:
                self.PHResolveTime[time].MIPSolvers[w].GivenQuantity = givenquantity
                self.PHResolveTime[time].MIPSolvers[w].CreateCopyGivenQuantityConstraints()
                self.PHResolveTime[time].MIPSolvers[w].GivenConsumption = givenconsumption
                self.PHResolveTime[time].MIPSolvers[w].CreateCopyGivenConumptionConstraints()

            self.IsDefinePHResolve[time] = True

        #Update the model for made decisions
        else:
            self.PHResolveTime[time].UpdateForDemand(demanduptotimet)
            #self.PHResolveTime[time].UpdateForSetup(givensetup)

            self.PHResolveTime[time].UpdateForQuantity(givenquantity)
            self.PHResolveTime[time].UpdateForConsumption(givenconsumption)

        #Re-set the parameters
        self.PHResolveTime[time].ReSetParameter()

        #solve.
        solution = self.PHResolveTime[time].Run()

        #get the result.
        qty = [solution.ProductionQuantity[0][time][p] for p in self.Instance.ProductSet]
        consumption = [solution.Consumption[0][time][c[0]][c[1]] for c in self.Instance.ConsumptionSet]

        #print the result.
        return qty, consumption


    # This function return the setup decision and quantity to produce for the scenario given in argument
    def GetDecisionFromSDDPForScenario(self, sddp, scenario):

        #Get the setup quantitities associated with the solultion
        givensetup = [[sddp.GetSetupFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet]
                      for t in self.Instance.TimeBucketSet]

        #givenquantty = [[sddp.GetQuantityFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet]
        #              for t in range(self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertainty)]
        givenquantty = [[0 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]
        givenconsumption = [[0 for c in range(len(self.Instance.ConsumptionSet))] for t in self.Instance.TimeBucketSet]


        #givenquantty=[]
        #Copy the quantity from the last stage
        for stage in sddp.ForwardStage:
            for t in stage.RangePeriodQty:
                time = stage.GetTimePeriodAssociatedToQuantityVariable(0, t)

                for p in self.Instance.ProductSet:
                    givenquantty[time][p] = stage.QuantityValues[scenario][t][p]

                for c in range(len(self.Instance.ConsumptionSet)):
                    givenconsumption[time][c] = stage.ConsumptionValues[scenario][t][c]

        return givensetup, givenquantty, givenconsumption

    def GetScenarioSet(self, solveseed, nrscenario, allscenarios):
        scenarioset = []
        treeset = []
        # Use an offset in the seed to make sure the scenario used for evaluation are different from the scenario used for optimization
        offset = solveseed + 999323

        #Uncoment to generate all the scenario if a  distribution with smallll support is used
        if allscenarios == 1:
            if Constants.Debug:
                    print("Generate all the scenarios")

            scenariotree = ScenarioTree(self.Instance, [1] + [1]*self.Instance.NrTimeBucketWithoutUncertaintyBefore + [8, 8, 8, 8, 0], offset,
                                         scenariogenerationmethod=Constants.All,
                                         model=Constants.ModelYFix)
            scenarioset = scenariotree.GetAllScenarios(False)

            for s in range(len(scenarioset)):
                tree = ScenarioTree(self.Instance, [1, 1, 1, 1, 1, 1, 1, 1, 0], offset,
                                    model=Constants.ModelYFix, givenfirstperiod=scenarioset[s].Demands)
                treeset.append(tree)
        else:
            for seed in range(offset, nrscenario + offset, 1):
                # Generate a random scenario
                ScenarioSeed = seed
                # Evaluate the solution on the scenario
                treestructure = [1] + [1] * self.Instance.NrTimeBucket + [0]

                scenariotree = ScenarioTree(self.Instance, treestructure, ScenarioSeed, evaluationscenario=True,
                                            scenariogenerationmethod="MC")
                scenario = scenariotree.GetAllScenarios(False)[0]

                scenarioset.append(scenario)
                treeset.append(scenariotree)


        return scenarioset, treeset

    def ComputeInformation(self, Evaluation, nrscenario):
        Sum = sum(Evaluation[s][sol] for s in range(nrscenario) for sol in range(self.NrSolutions))
        Average = Sum/nrscenario
        sumdeviation = sum(
            math.pow((Evaluation[s][sol] - Average), 2) for s in range(nrscenario) for sol in range(self.NrSolutions))
        std_dev = math.sqrt((sumdeviation / nrscenario))

        EvaluateInfo = [nrscenario, Average, std_dev]

        return EvaluateInfo

    def ComputeStatistic(self, Evaluated, Probabilities, KPIStat, nrerror):


        mean = float(sum(np.dot(Evaluated[k][m], Probabilities[k][m])
                         for k in range(len(Evaluated) )
                         for m  in range(self.EvalatorIdentificator.NrEvaluation) if Evaluated[k][m]>=0)
                     / sum(Probabilities[k][m]
                           for k in range(len(Evaluated))
                           for m in range(self.EvalatorIdentificator.NrEvaluation) if Evaluated[k][m] >= 0
                            if Evaluated[k]>=0))
        K = len(Evaluated)
        M = self.EvalatorIdentificator.NrEvaluation
        variancepondere = (1.0 / K) * \
                           sum(Probabilities[k][seed] * math.pow(Evaluated[k][seed]- mean, 2)
                               for seed in range(M)
                               for k in range(K))

        variance2 = ((1.0 / K) * sum((1.0 / M) * sum(math.pow(Evaluated[k][seed], 2) for seed in range(M)) for k in range(K))) - math.pow(mean,  2)
        covariance = 0

        for seed in range(M):
            step = 1
            for k in range(K):
                step *= (math.pow(Evaluated[k][seed] - mean, 2))
            covariance += Probabilities[0][seed] * 1/K * step

        term = stats.norm.ppf(1 - 0.05) * math.sqrt(max(((variancepondere + (covariance * (M - 1))) / (K * M)), 0.0))
        LB = K
        UB = -1
        d = datetime.now()
        date = d.strftime('%m_%d_%Y_%H_%M_%S')

        EvaluateInfo = self.ComputeInformation(Evaluated, self.EvalatorIdentificator.NrEvaluation)

        MinAverage = min((1.0 / M) * sum(Evaluated[k][seed] for seed in range(M)) for k in range(K))
        MaxAverage = max((1.0 / M) * sum(Evaluated[k][seed] for seed in range(M)) for k in range(K))

        if Constants.PrintDetailsExcelFiles:
            general = self.TestIdentificator.GetAsStringList() \
                      + self.EvalatorIdentificator.GetAsStringList() \
                      + [mean, variance2, covariance, LB, UB, MinAverage,
                         MaxAverage, nrerror]

            columnstab = ["Instance", "Distribution", "Model", "NrInSampleScenario", "Identificator", "Mean", "Variance",
                      "Covariance", "LB", "UB", "Min Average", "Max Average", "nrerror"]
            myfile = open(r'./Test/Bounds/TestResultOfEvaluated_%s_%r_%s_%s.csv' % (
                            self.Instance.InstanceName, self.EvalatorIdentificator.PolicyGeneration,
                            self.Model, date), 'wb')
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(general)
            myfile.close()

        KPIStat = KPIStat[6:] #The first values in KPIStats are not interesting for out of sample evalution (see MRPSolution::PrintStatistics)
        EvaluateInfo = [mean, LB, UB, MinAverage, MaxAverage, nrerror] + KPIStat

        return EvaluateInfo


    def EvaluateSDDPMethod(self):
        #Get the required number of scenario
        self.GetScenarioSet()

    #Recover the production quantity from the last MIP resolution
    def GetQuantityByResolve(self, demanduptotimet, resolvetime, givenquantty, givenconsumption, solution, givensetup):
        error = 0
        if resolvetime <= self.Instance.NrTimeBucketWithoutUncertaintyBefore:  # return the quantity at the root of the node
            resultqty = [solution.ProductionQuantity[0][resolvetime][p] for p in self.Instance.ProductSet]
            resultqtyconsumption = [solution.Consumption[0][resolvetime][c[0]][c[1]] for c in self.Instance.ConsumptionSet]
        else:
            quantitytofix = [[givenquantty[t][p] for p in self.Instance.ProductSet] for t in range(resolvetime)]
            consumptiontfix = [[givenconsumption[t][c] for c in range(len(self.Instance.ConsumptionSet))] for t in range(resolvetime)]


            if self.TestIdentificator.Method == Constants.MIP:
                    resultqty, resultqtyconsumption, error = self.ResolveMIP(quantitytofix, givensetup, consumptiontfix, demanduptotimet, resolvetime)
            if self.TestIdentificator.Method == Constants.ProgressiveHedging:
                    resultqty, resultqtyconsumption = self.GetDecisionFromPHForScenario(demanduptotimet, resolvetime, quantitytofix, consumptiontfix, givensetup)

        return resultqty, resultqtyconsumption, error

    def ResolveRule(self, quantitytofix,  givensetup, demanduptotimet, time):


        solution = self.DecentralizedMRP.SolveWithSimpleRule( self.Model, givensetup, quantitytofix, time-1, demanduptotimet)

        result = [solution.ProductionQuantity[0][time][p] for p in self.Instance.ProductSet]
        return result


    #This function generate the scenario tree for the next planning horizon
    def GetScenarioTreeForResolve(self, resolvetime, demanduptotimet):
        treestructure = [1] \
                        + [self.ReferenceTreeStructure[
                               t - (resolvetime - self.Instance.NrTimeBucketWithoutUncertaintyBefore) + 1]
                               if (t >= resolvetime and (t < (self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)))
                               else 1
                               for t in range(self.Instance.NrTimeBucket)] \
                        + [0]
        if self.Model == Constants.ModelYQFix:
            treestructure = [1] \
                            + [self.ReferenceTreeStructure[1]
                               if (t == resolvetime)
                               else 1
                               for t in range(self.Instance.NrTimeBucket)] \
                            + [0]

        if self.Model == Constants.ModelYQFix and self.ScenarioGenerationResolvePolicy == Constants.All:
            nrstochasticperiod = self.Instance.NrTimeBucket - resolvetime
            treestructure = [1] \
                            + [int(math.pow(8, nrstochasticperiod))
                               if (t == resolvetime and (
                        t < (self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)))
                               else 1
                               for t in range(self.Instance.NrTimeBucket)] \
                            + [0]

        # self.StartSeedResolve = self.StartSeedResolve + 1
        scenariotree = ScenarioTree(self.Instance, treestructure, self.StartSeedResolve,
                                    averagescenariotree=self.EvaluateAverage,
                                    givenfirstperiod=demanduptotimet,
                                    scenariogenerationmethod=self.ScenarioGenerationResolvePolicy,
                                    model=self.Model)

        return scenariotree, treestructure

    #Run the MIP with some decisions fixed in previous iterations
    def ResolveMIP(self, quantitytofix,  givensetup, consumptiontofix, demanduptotimet, resolvetime):
            if not self.IsDefineMIPResolveTime[resolvetime]:
                scenariotree, _ = self.GetScenarioTreeForResolve(resolvetime, demanduptotimet)

                mipsolver = MIPSolver(self.Instance, self.Model, scenariotree,
                                      self.EVPI,
                                      implicitnonanticipativity=(not self.EVPI),
                                      evaluatesolution=True,
                                      givenquantities=quantitytofix,
                                      givensetups=givensetup,
                                      givenconsumption=consumptiontofix,
                                      fixsolutionuntil=(resolvetime -1), #time lower or equal
                                      demandknownuntil=resolvetime,
                                      usesafetystock=self.UseSafetyStock,
                                      usesafetystockgrave=self.UseSafetyStockGrave)


                mipsolver.BuildModel()
                self.MIPResolveTime[resolvetime] = mipsolver
                self.IsDefineMIPResolveTime[resolvetime] = True
            else:

                self.MIPResolveTime[resolvetime].ModifyMipForScenario(demanduptotimet, resolvetime)
                self.MIPResolveTime[resolvetime].ModifyMipForFixQuantity(quantitytofix, fixuntil=resolvetime)
                self.MIPResolveTime[resolvetime].ModifyMipForFixConsumption(consumptiontofix, fixuntil=resolvetime)


            self.MIPResolveTime[resolvetime].Cplex.parameters.advance = 1
            self.MIPResolveTime[resolvetime].Cplex.parameters.lpmethod.set(self.MIPResolveTime[resolvetime].Cplex.parameters.lpmethod.values.barrier)

            solution = self.MIPResolveTime[resolvetime].Solve(createsolution=False)

            if Constants.Debug:
                print("End solving")


            #self.MIPResolveTime[time].Cplex.write("MRP-Re-Solve.lp")
            # Get the corresponding node:
            error = 0
            sol = self.MIPResolveTime[resolvetime].Cplex.solution
            if sol.is_primal_feasible():
                array = [self.MIPResolveTime[resolvetime].GetIndexQuantityVariable(p, resolvetime, 0) for p in self.Instance.ProductSet];

                resultqty = sol.get_values(array)
                if Constants.Debug:
                    print(resultqty)

                array = [int(self.MIPResolveTime[resolvetime].GetIndexConsumptionVariable(c[0], c[1], resolvetime, 0)) for c in self.Instance.ConsumptionSet];
                resultconsumption = sol.get_values(array)
                if Constants.Debug:
                    print(resultqty)
            else:
                if Constants.Debug:
                    self.MIPResolveTime[resolvetime].Cplex.write("MRP-Re-Solve.lp")
                    raise NameError("Infeasible MIP at time %d in Re-solve see MRP-Re-Solve.lp" % resolvetime)

                error = 1

            return resultqty, resultconsumption, error

            # Create the set of subinstance to solve in a rolling horizon approach

