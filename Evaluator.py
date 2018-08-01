from __future__ import absolute_import, division, print_function
from Constants import Constants
from Solution import Solution
from EvaluatorIdentificator import EvaluatorIdentificator
from EvaluationSimulator import EvaluationSimulator
from SDDP import SDDP
import subprocess
import cPickle as pickle
import csv
import datetime

#This class contains the method to call the simulator and run the evauation
class Evaluator( object ):

    # Constructor
    def __init__(self, instance, testidentifier, evaluatoridentificator, solver):
        self.TestIdentifier = testidentifier
        self.EvalutorIdentificator = evaluatoridentificator
        self.Solutions = self.GetPreviouslyFoundSolution()
        self.Instance = instance
        self.Solver = solver
        self.OutOfSampleTestResult = []
        self.InSampleTestResult =[]
        if self.TestIdentifier.Method == Constants.SDDP:
            if Constants.SDDPSaveInExcel:
                self.Solver.SDDPSolver = SDDP(instance, self.TestIdentifier)
                self.Solver.SDDPSolver.LoadCuts()
            Constants.SDDPGenerateCutWith2Stage = False
            Constants.SDDPRunSigleTree = False
            Constants.SolveRelaxationFirst = False

    #Return the solution to evaluate
    def GetPreviouslyFoundSolution(self):
        result = []
        seeds = [self.TestIdentifier.ScenarioSeed]
        for s in seeds:
            try:
                self.TestIdentifier.ScenarioSeed = s
                filedescription = self.TestIdentifier.GetAsString()
                solution = Solution()
                solution.ReadFromFile(filedescription)
                result.append(solution)

            except IOError:
                if Constants.Debug:
                    print(IOError)
                    print("No solution found for seed %d" % s)

        return result

    # return a set of statistic associated with solving the problem
    def ComputeInSampleStatistis(self):
        InSampleKPIStat = []

        solutions = self.GetPreviouslyFoundSolution()
        lengthinsamplekpi = -1

        for solution in solutions:
            if not Constants.PrintOnlyFirstStageDecision:
                solution.ComputeStatistics()
            insamplekpisstate = solution.PrintStatistics(self.TestIdentifier,
                                                         "InSample",-1, 0, -1, True,
                                                         self.TestIdentifier.ScenarioSampling)
            lengthinsamplekpi = len(insamplekpisstate)
            InSampleKPIStat = [0] * lengthinsamplekpi
            for i in range(lengthinsamplekpi):
                InSampleKPIStat[i] = InSampleKPIStat[i] + insamplekpisstate[i]

        for i in range(lengthinsamplekpi):
            InSampleKPIStat[i] = InSampleKPIStat[i] / len(solutions)

        return InSampleKPIStat

    #run the evaluation
    def Evaluate(self):
        self.InSampleTestResult = self.ComputeInSampleStatistis()

        solutions = self.GetPreviouslyFoundSolution()
        evaluator = Evaluator(self.Instance, solutions, [self.Solver.SDDPSolver], self.TestIdentifier, self.EvaluatorIdentifier,
                              self.GetTreeStructure())
        self.OutOfSampleTestResult = evaluator.EvaluateYQFixSolution(self.TestIdentifier, self.EvaluatorIdentifier)

    #Get the temporary file conting the results of the simulation
    def GetEvaluationFileName(self):

        result = Constants.EvaluationFolder + self.TestIdentifier.GetAsString() + self.EvalutorIdentificator.GetAsString()
        return result

    #run the simulation
    def EvaluateSingleSol(self):
        tmpmodel = self.TestIdentifier.Model
        filedescription = self.TestIdentifier.GetAsString()

        MIPModel = self.TestIdentifier.Model
        if Constants.IsDeterministic(self.TestIdentifier.Model):
            MIPModel = Constants.ModelYQFix
        if self.TestIdentifier.Model == Constants.ModelHeuristicYFix:
            MIPModel = Constants.ModelYFix
            self.TestIdentifier.Model = Constants.ModelYFix

        solution = Solution()
        if not self.TestIdentifier.EVPI and not self.TestIdentifier.ScenarioSampling == Constants.RollingHorizon:
            #In evpi mode, a solution is computed for each scenario
            if Constants.RunEvaluationInSeparatedJob:
                solution.ReadFromFile(filedescription)
            else:
                solution = self.GetPreviouslyFoundSolution()[0]

                if not solution.IsPartialSolution:
                    solution.ComputeCost()

                    if self.TestIdentifier.Model <> Constants.ModelYQFix:
                        solution.ScenarioTree.FillQuantityToOrderFromMRPSolution(solution)

        evaluator = EvaluationSimulator(self.Instance, [solution], [self.Solver.SDDPSolver],
                                        testidentificator=self.TestIdentifier,
                                        evaluatoridentificator=self.EvalutorIdentificator,
                                        treestructure=self.Solver.GetTreeStructure(),
                                        model=MIPModel)

        self.OutOfSampleTestResult = evaluator.EvaluateYQFixSolution(saveevaluatetab=True,
                                                                     filename=self.GetEvaluationFileName(),
                                                                     evpi=self.TestIdentifier.EVPI)

        self.TestIdentifier.Model = tmpmodel
        self.GatherEvaluation()

    def GatherEvaluation(self):

        evaluator = EvaluationSimulator(self.Instance, solutions=[], sddps=[],
                                        testidentificator=self.TestIdentifier,
                                        evaluatoridentificator=self.EvalutorIdentificator,
                                        treestructure=self.Solver.GetTreeStructure())
        EvaluationTab = []
        ProbabilitiesTab =[]
        KPIStats = []
        nrfile = 0
        #Creat the evaluation table
        currentseedvalue = self.TestIdentifier.ScenarioSeed
        for seed in [self.TestIdentifier.ScenarioSeed]:#SeedArray:
            try:
                filename = self.GetEvaluationFileName()
                self.TestIdentifier.ScenarioSeed = seed
                #print "open file %rEvaluator.txt"%filename
                with open(filename + "Evaluator.txt", 'rb') as f:
                    list = pickle.load(f)
                    EvaluationTab.append(list)

                with open(filename + "Probabilities.txt", 'rb') as f:
                    list = pickle.load(f)
                    ProbabilitiesTab.append(list)

                with open(filename + "KPIStat.txt", "rb") as f:  # Pickling
                    list = pickle.load(f)
                    KPIStats.append(list)
                    nrfile =nrfile +1
            except IOError:
                if Constants.Debug:
                    print("No evaluation file found for seed %d" % seed)

        if nrfile >= 1:
            KPIStat = [sum(e) / len(e) for e in zip(*KPIStats)]

            self.OutOfSampleTestResult = evaluator.ComputeStatistic(EvaluationTab, ProbabilitiesTab, KPIStat, -1)
            if self.TestIdentifier.Method == Constants.MIP and not self.TestIdentifier.EVPI:
                self.InSampleTestResult = self.ComputeInSampleStatistis()
            self.InSampleTestResult = self.ComputeInSampleStatistis()
            self.PrintFinalResult()

        self.TestIdentifier.ScenarioSeed = currentseedvalue

    def PrintFinalResult(self):
        data = self.TestIdentifier.GetAsStringList() +\
               self.EvalutorIdentificator.GetAsStringList() +\
               self.InSampleTestResult + self.OutOfSampleTestResult
        #d = datetime.now()
        #date = d.strftime('%m_%d_%Y_%H_%M_%S')
        if Constants.Debug:
            print("print the test result ./Test/TestResult_%s_%s.csv" % (
                self.TestIdentifier.GetAsString(), self.EvalutorIdentificator.GetAsString()))
        myfile = open(r'./Test/TestResult_%s_%s.csv' % (self.TestIdentifier.GetAsString(),
                                                        self.EvalutorIdentificator.GetAsString()), 'wb')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(data)
        myfile.close()

    #This function runs the evaluation for the just completed test :
    def RunEvaluation(self):
        if Constants.LauchEvalAfterSolve:
            policyset = ["S", "Re-solve"]

            if self.TestIdentifier.NrScenario == "6400b" or self.TestIdentifier.Method == Constants.SDDP or self.TestIdentifier.Method == Constants.ProgressiveHedging:
                policyset = ["Re-solve"]

            if self.TestIdentifier.NrScenario == "6400c":
                policyset = ["S"]

            if self.TestIdentifier.Model == Constants.ModelYQFix \
                    or Constants.IsDeterministic(self.TestIdentifier.Model)\
                    or Constants.IsRule(self.TestIdentifier.Model):
                    policyset = ["Fix", "Re-solve"]

            if self.Instance.NrTimeBucket >= 10 and not self.TestIdentifier.Model == Constants.ModelHeuristicYFix:
                policyset = ["Fix"]

            perfectsenarioset = [0]
            if self.Instance.Distribution == Constants.Binomial:
                perfectsenarioset = [0, 1]
            for policy in policyset:
                for perfectset in perfectsenarioset:
                    if Constants.RunEvaluationInSeparatedJob:
                        jobname = "./Jobs/job_evaluate_%s_%s_%s_%s_%s_%s_%s" % (
                            self.TestIdentifier.InstanceName,
                            self.TestIdentifier.Model,
                            self.TestIdentifier.NrScenario,
                            self.TestIdentifier.ScenarioSampling,
                            self.TestIdentifier.Method,
                            policy,
                            self.TestIdentifier.ScenarioSeed)
                        subprocess.call(["qsub", jobname])
                    else:
                        PolicyGeneration = policy
                        NearestNeighborStrategy = policy
                        AllScenario = perfectset
                        if AllScenario == 1:
                            NrEvaluation = 4096
                        else:
                            NrEvaluation = self.EvalutorIdentificator.NrEvaluation

                        self.EvalutorIdentificator = EvaluatorIdentificator(policy,
                                                                            NrEvaluation,
                                                                            self.EvalutorIdentificator.TimeHorizon,
                                                                            perfectset)

                        self.EvaluateSingleSol()
