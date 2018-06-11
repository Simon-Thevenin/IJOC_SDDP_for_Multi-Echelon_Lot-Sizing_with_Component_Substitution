from __future__ import absolute_import, division, print_function

from Constants import Constants
from Solution import Solution
from SDDPStage import SDDPStage
from SDDPLastStage import SDDPLastStage
from ScenarioTree import ScenarioTree
from MIPSolver import MIPSolver
from SDDPCallBack import SDDPCallBack
import pickle
from SDDPUserCutCallBack import SDDPUserCutCallBack

import numpy as np
import math
import time
import copy

# This class contains the attributes and methods allowing to define the SDDP algorithm.
class SDDP(object):

    #return the object stage associated with the decision stage given in paramter
    def GetSDDPStage(self, decisionstage):
        result = None
        if decisionstage >= 0:
            result = self.Stage[decisionstage]
        return result

    #Fill the links predecessor and next of each object stage
    def LinkStages(self):
        previousstage = None
        for stage in self.Stage:
            stage.PreviousSDDPStage = previousstage
            if not previousstage is None:
                previousstage.NextSDDPStage = stage
            previousstage = stage

    def __init__(self, instance, testidentifier):
        self.Instance = instance
        self.TestIdentifier = testidentifier
        nrstage = self.Instance.NrTimeBucket# - self.Instance.NrTimeBucketWithoutUncertainty
        self.StagesSet = range(nrstage + 1)
        self.CurrentIteration = 0
        self.CurrentLowerBound = 0
        self.BestUpperBound = Constants.Infinity
        self.CurrentUpperBound = Constants.Infinity
        self.VarianceForwardPass = -1
        self.StartOfAlsorithm = time.time()
        self.CurrentSetOfTrialScenarios = []
        self.SetOfSAAScenario = []
        self.SAAScenarioNrSet = []
        self.TrialScenarioNrSet = []
        self.IsIterationWithConvergenceTest = False
        self.CurrentScenarioSeed = int(self.TestIdentifier.ScenarioSeed)
        self.StartingSeed = self.TestIdentifier.ScenarioSeed
        self.Stage = [SDDPStage(owner=self, decisionstage=t) for t in range(nrstage)] \
                        + [SDDPLastStage(owner=self, decisionstage=nrstage)]
        self.LinkStages()
        self.CurrentNrScenario = Constants.SDDPNrScenarioForwardPass
        self.NrScenarioSAA = int(self.TestIdentifier.NrScenario)

        self.CurrentBigM = []
        self.ScenarioGenerationMethod = self.TestIdentifier.ScenarioSampling
        self.CurrentExpvalueUpperBound = Constants.Infinity
        self.EvaluationMode = False
        self.UseCorePoint = False
        self.GenerateStrongCut = Constants.GenerateStrongCut
        self.TraceFile = None
        self.TraceFileName = "./Temp/trace_%s_%s.txt" % (self.Instance.InstanceName, self.TestIdentifier.MIPSetting)
        self.HeuristicSetupValue = []



    #This function make the forward pass of SDDP
    def ForwardPass(self, ignorefirststage = False):
        if Constants.Debug:
            print("Start forward pass")
        self.SetCurrentBigM()
        for t in self.StagesSet:
            if not ignorefirststage or t >= 1:

                if Constants.SDDPCleanCuts \
                    and self.CurrentIteration > 0 \
                    and self.CurrentIteration % 100 == 0 \
                    and (t >= 1 or not Constants.SDDPRunSigleTree):
                        self.Stage[t].CleanCuts()

                #Run the forward pass at each stage t
                self.Stage[t].RunForwardPassMIP()

                #try to use core point method, remove if it does not work
                #if self.Stage[t].IsFirstStage()



    #This function make the backward pass of SDDP
    def BackwardPass(self, returnfirststagecut=False):
        if Constants.Debug:
            print("Start Backward pass")

        self.UseCorePoint = self.GenerateStrongCut

        for t in self.StagesSet:
            if self.GenerateStrongCut:
                self.Stage[t].UpdateCorePoint()

        #if self.IsIterationWithConvergenceTest:
        #    self.ConsideredTrialForBackward = np.random.randint(self.CurrentNrScenario, size=Constants.SDDPNrScenarioBackwardPass)
        #else:
        self.ConsideredTrialForBackward = self.TrialScenarioNrSet

        #self.ConsideredTrialForBackward = np.random.randint(self.CurrentNrScenario,
        #                                                    size=Constants.SDDPNrScenarioBackwardPass)


        #generate a cut for each trial solution
        for t in reversed(range(1, len(self.StagesSet))):
            #Build or update the MIP of stage t
            returncutinsteadofadd = (returnfirststagecut and t == 1)
            firststagecuts, avgsubprobcosts = self.Stage[t].GernerateCut(returncutinsteadofadd)

        self.UseCorePoint = False

        if returnfirststagecut:
            return firststagecuts, avgsubprobcosts


    def GenerateScenarios(self, nrscenario, average = False):
        if Constants.Debug:
            print("Start generation of new scenarios")

        #Generate a scenario tree
        treestructure = [1, nrscenario] + [1] * (self.Instance.NrTimeBucket - 1) + [0]

        scenariotree = ScenarioTree(self.Instance, treestructure, self.CurrentScenarioSeed,
                                    scenariogenerationmethod=self.ScenarioGenerationMethod,
                                    generateasYQfix=False, averagescenariotree=average,
                                    model=Constants.ModelYQFix)



        #Get the set of scenarios
        scenarioset = scenariotree.GetAllScenarios(computeindex=False)





        return scenarioset

    #This function generates the scenarios for the current iteration of the algorithm
    def GenerateTrialScenarios(self):

         #self.CurrentIteration % 25 == 0

        if self.IsIterationWithConvergenceTest:
            self.CurrentNrScenario = Constants.SDDPNrScenarioTest
        else:
            self.CurrentNrScenario = Constants.SDDPNrScenarioForwardPass
        self.CurrentSetOfTrialScenarios = self.GenerateScenarios(self.CurrentNrScenario, average=Constants.SDDPDebugSolveAverage)
        self.TrialScenarioNrSet = range(len(self.CurrentSetOfTrialScenarios))
        self.CurrentNrScenario = len(self.CurrentSetOfTrialScenarios)
        #Modify the number of scenario at each stage
        for stage in self.StagesSet:
            self.Stage[stage].SetNrTrialScenario(self.CurrentNrScenario)

        self.CurrentScenarioSeed = self.CurrentScenarioSeed + 1


    #This function generates the scenarios for the current iteration of the algorithm
    def GenerateSAAScenarios(self):

        self.SetOfSAAScenario = self.GenerateScenarios(self.NrScenarioSAA, average=Constants.SDDPDebugSolveAverage)
        self.SAAScenarioNrSet = range(len(self.SetOfSAAScenario))
        self.NrScenarioSAA = len(self.SetOfSAAScenario)
        #Modify the number of scenario at each stage
        #for stage in self.StagesSet:
        #    self.Stage[stage].SAAScenarioNrSet(len(self.CurrentSetOfTrialScenarios))

        self.CurrentScenarioSeed = self.CurrentScenarioSeed + 1


    #This function return the quanity of product to produce at time which has been decided at an earlier stage
    def GetQuantityFixedEarlier(self, product, time, scenario):
        if self.UseCorePoint:
            result = self.Stage[time].CorePointQuantityValues[scenario][product]
        else:
            result = self.Stage[time].QuantityValues[scenario][product]

        return result

    # This function return the inventory  of product  at time which has been decided at an earlier stage
    def GetInventoryFixedEarlier(self, product, time, scenario):
        if time == -1:
            result = self.Instance.StartingInventories[product]
        else:
            decisiontime = 0
            if self.Instance.HasExternalDemand[product]:
                decisiontime = time + 1
            else:
                decisiontime = time

            if self.UseCorePoint:
                result = self.Stage[decisiontime].CorePointInventoryValue[scenario][product]
            else:
                result = self.Stage[decisiontime].InventoryValue[scenario][product]

        return result

    # This function return the backordered quantity of product which has been decided at an earlier stage
    def GetBackorderFixedEarlier(self, product, time, scenario):
        if time == -1:
            result = 0
        else:
            if self.UseCorePoint:
                result = self.Stage[time + 1].CorePointBackorderValue[scenario][self.Instance.ProductWithExternalDemandIndex[product]]
            else:
                result = self.Stage[time + 1].BackorderValue[scenario][self.Instance.ProductWithExternalDemandIndex[product]]

        return result

    #This function return the value of the setup variable of product to produce at time which has been decided at an earlier stage
    def GetSetupFixedEarlier(self, product, time, scenario):
        if self.UseCorePoint:
            result = self.Stage[0].CorePointProductionValue[scenario][time][product]
        else:
            result = self.Stage[0].ProductionValue[scenario][time][product]
        return result

    #This function return the demand of product at time in scenario
    def GetDemandQuantity(self, product, time, scenario):
        result = 0
        return result

    #This funciton update the lower bound based on the last forward pass
    def UpdateLowerBound(self):
        result = self.Stage[0].PassCostWithAproxCosttoGo
        self.CurrentLowerBound = result

    #This funciton update the upper bound based on the last forward pass
    def UpdateUpperBound(self):
            laststage = len(self.StagesSet) - 1
#            expectedupperbound = sum( self.Stage[ s ].PassCost for s in self.StagesSet )
#            variance = math.pow( np.std(  [ sum( self.Stage[ s ].PartialCostPerScenario[w] for s in self.StagesSet ) for w in self.ScenarioNrSet]  ), 2 )
            expectedupperbound = self.Stage[laststage].PassCost
            variance = sum(math.pow(expectedupperbound-self.Stage[laststage].PartialCostPerScenario[w], 2) for w in self.TrialScenarioNrSet) / self.CurrentNrScenario
            self.CurrentUpperBound = expectedupperbound - 1.96 * math.sqrt(variance / self.CurrentNrScenario)
            self.CurrentExpvalueUpperBound = expectedupperbound
            self.VarianceForwardPass = variance

    #This function check if the stopping criterion of the algorithm is met
    def CheckStoppingCriterion(self):
        duration = time.time() - self.StartOfAlsorithm
        timalimiteached = (duration > Constants.AlgorithmTimeLimit)
        optimalitygap = ( self.CurrentUpperBound - self.CurrentLowerBound)/self.CurrentUpperBound
        convergencecriterion = 1000
        if self.CurrentLowerBound > 0:
            convergencecriterion = float(self.CurrentUpperBound) / float(self.CurrentLowerBound)

        delta = 1000
        if self.VarianceForwardPass > 0 and self.CurrentLowerBound> 0:
            delta = 3.92 * math.sqrt(float(self.VarianceForwardPass) / float(self.CurrentNrScenario)) \
                    /  float(self.CurrentLowerBound)

        convergencecriterionreached = convergencecriterion < 1 \
                                      and delta < Constants.AlgorithmOptimalityTolerence



        optimalitygapreached = (optimalitygap < Constants.AlgorithmOptimalityTolerence)
        iterationlimitreached = (self.CurrentIteration > Constants.SDDPIterationLimit)
        result = self.IsIterationWithConvergenceTest and convergencecriterionreached or timalimiteached or iterationlimitreached
        if Constants.PrintSDDPTrace:
            self.WriteInTraceFile("Iteration: %d, Duration: %d, LB: %r, UB: %r (exp:%r), convtets:%r Gap: %r, conv: %r, delta : %r \n"
                                  %(self.CurrentIteration, duration, self.CurrentLowerBound, self.CurrentUpperBound,
                                    self.CurrentExpvalueUpperBound, self.IsIterationWithConvergenceTest,
                                    optimalitygap, convergencecriterion, delta))

        if result == False and self.CurrentLowerBound > self.CurrentExpvalueUpperBound\
                and not self.IsIterationWithConvergenceTest:
            self.IsIterationWithConvergenceTest = True
            self.GenerateTrialScenarios()
            self.ForwardPass()
            self.ComputeCost()
            self.UpdateUpperBound()
            result = self.CheckStoppingCriterion()

        return result

    def CheckStoppingRelaxationCriterion(self, round):

        optimalitygap = (self.CurrentExpvalueUpperBound - self.CurrentLowerBound)/self.CurrentExpvalueUpperBound
        optimalitygapreached = (optimalitygap < Constants.SDDPGapRelax)
        iterationlimitreached = (self.CurrentIteration > Constants.SDDPNrIterationRelax * round)
        result = optimalitygapreached or iterationlimitreached
        return result

    #This funciton compute the solution of the scenario given in argument (used after to have run the algorithm, and the cost to go approximation are built)
    def ComputeSolutionForScenario(self, scenario):
        solution = Solution()
        return solution

    def WriteInTraceFile(self, string):
        self.TraceFile = open(self.TraceFileName, "a")
        self.TraceFile.write(string)
        self.TraceFile.close()

    #This function runs the SDDP algorithm
    def Run(self):
        if Constants.PrintSDDPTrace:
            self.TraceFile = open(self.TraceFileName, "w")

            self.TraceFile.write("Start the SDDP algorithm \n")
            self.TraceFile.write("Use Papadakos method to generate strong cuts: %r \n"%Constants.GenerateStrongCut)
            self.TraceFile.write("Generate a 10000 cuts with linear relaxation: %r \n"%Constants.SolveRelaxationFirst)
            self.TraceFile.write("Use valid inequalities: %r \n"%Constants.SDDPUseValidInequalities)
            self.TraceFile.write("Run SDDP in a single tree: %r \n"%Constants.SDDPRunSigleTree)
            self.TraceFile.close()
        self.StartOfAlsorithm = time.time()
       # print("Attention SDDP solve average")

        self.GenerateSAAScenarios()
        round = 1
        createpreliminarycuts = Constants.SolveRelaxationFirst
        ExitLoop = not createpreliminarycuts and Constants.SDDPRunSigleTree

        while (not self.CheckStoppingCriterion() or createpreliminarycuts) and not ExitLoop:

            if createpreliminarycuts and self.CheckStoppingRelaxationCriterion(round):
                round += 1
                #if round < 3:
                #    self.Stage[0].ChangeSetupToValueOfTwoStage()
                #    self.WriteInTraceFile("Change stage 1 problem to heuristic solution \n")
                #elif round == 3:
                #    print("round3")
                ExitLoop = Constants.SDDPRunSigleTree
                #    print(ExitLoop)
                createpreliminarycuts = False
                self.CurrentUpperBound = Constants.Infinity
                self.CurrentLowerBound = 0
                if not ExitLoop:
                    self.Stage[0].ChangeSetupToBinary()
                    self.WriteInTraceFile("Change stage 1 problem to integer \n")

            #print("Attention uncomment")
            self.IsIterationWithConvergenceTest = False
            self.GenerateTrialScenarios()
            if Constants.Debug:
                print("********************Scenarios*********************")
                for s in self.CurrentSetOfTrialScenarios:
                    print("Demands: %r" % s.Demands)
                print("****************************************************")

            self.ForwardPass()
            self.ComputeCost()
            self.UpdateLowerBound()
            self.UpdateUpperBound()
            self.BackwardPass()
            self.CurrentIteration = self.CurrentIteration + 1
            #if self.CurrentIteration % 50 == 0:
            #    self.CurrentNrScenario =  self.CurrentNrScenario  * 10

            if self.CurrentIteration % 100 < 5:
                solution = self.CreateSolutionOfScenario(0)
                solution.PrintToExcel("solution_at_iteration_%r"%self.CurrentIteration)

        if Constants.SDDPRunSigleTree:
            self.RunSingleTreeSDDP()

        self.RecordSolveInfo()
        if Constants.PrintSDDPTrace:
            self.WriteInTraceFile("End of the SDDP algorithm\n ")


    # This function runs the SDDP algorithm
    def RunSingleTreeSDDP(self):

        if not Constants.SolveRelaxationFirst:
            #run forward pass to create the MIPS
            Constants.SolveRelaxationFirst = True
            self.ForwardPass()
            Constants.SolveRelaxationFirst = False

        self.CopyFirstStage = SDDPStage(owner=self, decisionstage=0)
        self.CopyFirstStage.SetNrTrialScenario(len(self.CurrentSetOfTrialScenarios))
        for cut in self.Stage[0].SDDPCuts:
            cut.Stage = None

        self.CopyFirstStage.SDDPCuts = copy.deepcopy(self.Stage[0].SDDPCuts)

        for cut in self.Stage[0].SDDPCuts:
            cut.Stage = self.Stage[0]

        self.CopyFirstStage.DefineMIP(0)

        for cut in self.CopyFirstStage.SDDPCuts:
            cut.Stage = self.CopyFirstStage
            cut.AddCut()


        self.CopyFirstStage.ChangeSetupToBinary()

        model_lazy = self.CopyFirstStage.Cplex.register_callback(SDDPCallBack)
        model_lazy.SDDPOwner = self
        model_lazy.Model = self.CopyFirstStage
       # model_usercut = self.CopyFirstStage.Cplex.register_callback(SDDPUserCutCallBack)
       # model_usercut.SDDPOwner = self
       # model_usercut.Model = self.CopyFirstStage

        self.CopyFirstStage.Cplex.write("./Temp/MainModel.lp")
        cplexlogfilename = "./Temp/CPLEXLog_%s_%s.txt"%(self.Instance.InstanceName, self.TestIdentifier.MIPSetting)
        self.CopyFirstStage.Cplex.set_log_stream(cplexlogfilename)
        self.CopyFirstStage.Cplex.set_results_stream(cplexlogfilename)
        self.CopyFirstStage.Cplex.set_warning_stream(cplexlogfilename)
        self.CopyFirstStage.Cplex.set_error_stream(cplexlogfilename)
        if Constants.Debug:
            print("Start To solve the main tree")
        self.CopyFirstStage.Cplex.solve()

        self.WriteInTraceFile("End Solve in one tree cost: %r " % self.CopyFirstStage.Cplex.solution.get_objective_value())

    def ComputeCost(self):
        for stage in self.Stage:
            stage.ComputePassCost()
            #stage.PassCost = sum( stage.StageCostPerScenario[w] for w in self.ScenarioNrSet  ) / len(self.ScenarioNrSet)

    def SetCurrentBigM(self):
       self.CurrentBigM =[MIPSolver.GetBigMValue(self.Instance, self.CurrentSetOfTrialScenarios, p) for p in self.Instance.ProductSet]

    def GetFileName(self):
        result ="./Solutions/SDDP_%s_%s_%s_Solution.sddp"%(self.Instance.InstanceName, self.StartingSeed, self.CurrentNrScenario)
        return result

#    def SaveInFile(self):
#        filename = self.GetFileName()
#        with open(filename , "w+") as fp:
#            pickle.dump(self.Stage[0].SDDPCuts[0], fp)

#    def ReadFromFile(self):
#        filename = self.GetFileName()
#        with open(filename , "rb") as fp:
#            pickle.load(self, fp)

    def RecordSolveInfo(self):
        self.SolveInfo = [self.Instance.InstanceName,
                          "SDDP",
                          self.CurrentUpperBound,
                          self.CurrentLowerBound,
                          self.CurrentIteration,
                          time.time() - self.StartOfAlsorithm,
                          0,
                          0,
                          #sol.progress.get_num_iterations(),
                          #sol.progress.get_num_nodes_processed(),
                          0,
                          0,
                          0,
                          0,
                          0,
                          0,
                          0,
                          self.Instance.NrLevel,
                          self.Instance.NrProduct,
                          self.Instance.NrTimeBucket,
                          0,
                          self.CurrentNrScenario,
                          self.Instance.MaxLeadTime,
                          0]

    def CreateSolutionAtFirstStage(self):
           # Get the setup quantitities associated with the solultion
            solproduction = [[[self.GetSetupFixedEarlier(p, t, 0) for p in self.Instance.ProductSet]
                          for t in self.Instance.TimeBucketSet]]

            solquantity = [[[self.GetQuantityFixedEarlier(p, 0, 0) for p in self.Instance.ProductSet]]]

            solinventory = [[[self.GetInventoryFixedEarlier(p, 0, 0)
                            if not self.Instance.HasExternalDemand[p]
                            else -1
                            for p in self.Instance.ProductSet]]]

            solbackorder = [[[-1 for p in self.Instance.ProductSet]]]

            solconsumption = [[[[-1 for p in self.Instance.ProductSet] for p in self.Instance.ProductSet]]]
            k=-1
            for c in self.Instance.ConsumptionSet:
                 k += 1
                 solconsumption[0][0][c[0]][c[1]] = self.Stage[0].ConsumptionValues[0][k]

            emptyscenariotree = ScenarioTree(instance=self.Instance,
                                             branchperlevel=[0,0,0,0,0],
                                             seed=self.TestIdentifier.ScenarioSeed)# instance = None, branchperlevel = [], seed = -1, mipsolver = None, evaluationscenario = False, averagescenariotree = False,  givenfirstperiod = [], scenariogenerationmethod = "MC", generateasYQfix = False, model = "YFix", CopyscenariofromYFIX=False ):


            solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder, solconsumption,
                                [0],  emptyscenariotree, partialsolution=True)

            solution.IsSDDPSolution = True
            solution.FixedQuantity = [[-1 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

            return solution


    def CreateSolutionOfScenario(self, scenario):
           # Get the setup quantitities associated with the solultion
            solproduction = [[[self.GetSetupFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet]
                          for t in self.Instance.TimeBucketSet]]

            solquantity = [[[self.GetQuantityFixedEarlier(p, t, scenario) for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]]

            solinventory = [[[self.GetInventoryFixedEarlier(p, t, scenario)

                            for p in self.Instance.ProductSet]
                             for t in self.Instance.TimeBucketSet]]

            solbackorder = [[[self.GetBackorderFixedEarlier(p, t, scenario)

                            for p in self.Instance.ProductSet]
                             for t in self.Instance.TimeBucketSet]]

            solconsumption = [[[[-1 for p in self.Instance.ProductSet] for p in self.Instance.ProductSet]for t in self.Instance.TimeBucketSet]]

            for t in self.Instance.TimeBucketSet:
                k = -1
                for c in self.Instance.ConsumptionSet:
                     k += 1
                     solconsumption[scenario][t][c[0]][c[1]] = self.Stage[t].ConsumptionValues[scenario][k]

            emptyscenariotree = ScenarioTree(instance=self.Instance,
                                             branchperlevel=[0,0,0,0,0],
                                             seed=self.TestIdentifier.ScenarioSeed)# instance = None, branchperlevel = [], seed = -1, mipsolver = None, evaluationscenario = False, averagescenariotree = False,  givenfirstperiod = [], scenariogenerationmethod = "MC", generateasYQfix = False, model = "YFix", CopyscenariofromYFIX=False ):


            solution = Solution(self.Instance, solquantity, solproduction, solinventory, solbackorder, solconsumption,
                                [self.CurrentSetOfTrialScenarios[0]], emptyscenariotree, partialsolution=False)

            solution.IsSDDPSolution = False
            solution.FixedQuantity = [[-1 for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

            return solution

    def SaveSolver(self):
        cuts = [[] for _ in self.StagesSet]
        for t in self.StagesSet:
            for cut in self.Stage[t].SDDPCuts:
                cut.Stage = None
                cuts[t].append(cut)
        with open("./Solutions/SDDP_%r.pkl"%self.TestIdentifier.GetAsStringList(), 'wb') as output:
            pickle.dump(cuts, output)

    def LoadCuts(self):

        with open("./Solutions/SDDP_%r.pkl"%self.TestIdentifier.GetAsStringList(), 'rb') as input:
            cuts = pickle.load(input)

        for t in self.StagesSet:
            for cut in cuts[t]:
                cut.Stage = self.Stage[t]
                self.Stage[t].SDDPCuts.append(cut)
