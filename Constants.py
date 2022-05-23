#This Class define a set of constant/parameters used in the rest of the code.
from __future__ import absolute_import, division, print_function
class Constants( object ):
    PathInstances = "./Instances/"
    PathCPLEXLog = "./CPLEXLog/"


    #Scenario sampling methods:
    MonteCarlo = "MC"
    QMC = "QMC"
    RQMC = "RQMC"
    All = "all"

    #Method
    MIP = "MIP"
    SDDP = "SDDP"
    ProgressiveHedging = "PH"
    Hybrid = "Hybrid"

    #Demand distributions:
    Lumpy = "Lumpy"
    SlowMoving = "SlowMoving"
    Uniform = "Uniform"
    Binomial = "Binomial"
    NonStationary = "NonStationary"

    #Methods
    Average = "Average"
    AverageSS = "AverageSS"
    AverageSSStat = "AverageSSStat"
    AverageSSDyn = "AverageSSStatDyn"
    AverageSSGrave = "AverageSSGrave"
    ModelYQFix = "YQFix"
    ModelYFix = "YFix"
    ModelHeuristicYFix = "HeuristicYFix"
    MLLocalSearch = "MLLocalSearch"
    JustYFix = "JustYFix"
    #Action
    Solve ="Solve"
    Evaluate = "Evaluate"

    #Decision Framework
    RollingHorizon = "RH"
    Fix = "Fix"
    Resolve = "Re-solve"

    #The set of seeds used for random number generator
    SeedArray = [788]#[2934, 875, 3545, 765, 546, 768, 242, 375, 142, 236, 788]
    EvaluationScenarioSeed = 2934

    #Running option
    Debug = False
    PrintSolutionFileToExcel = False
    PrintDetailsExcelFiles = False
    PrintOnlyFirstStageDecision = True
    RunEvaluationInSeparatedJob = False
    PrintScenarios = False
    PrintSolutionFileInTMP = False
    LauchEvalAfterSolve = True

    #Code parameter
    Infinity = 9999999999999
    AlgorithmTimeLimit = 7200
    MIPBasedOnSymetricTree = False
    RQMCAggregate = False

    #SDDPparameters
    AlgorithmOptimalityTolerence = 0.01#0.0005
    SDDPIterationLimit = 500
    SDDPPrintDebugLPFiles = False
    PrintSDDPTrace = True
    GenerateStrongCut = True
    SDDPRunSigleTree = False
    SDDPUseMultiCut = True

    SDDPModifyBackwardScenarioAtEachIteration = False

    #SDDPNrScenarioForwardPass = 10
    #SDDPNrScenarioBackwardPass = 10
    SDDPForwardPassInSAATree = False
    SDDPPerformConvergenceTestDuringRun = False
    SDDPIncreaseNrScenarioTest = 100
    SDDPInitNrScenarioTest = 1000

    SolveRelaxationFirst = True
    SDDPNrIterationRelax = 500
    SDDPGapRelax = 0.01
    SDDPUseValidInequalities = False
    SDDPGenerateCutWith2Stage = False
    SDDPCleanCuts = False
    SDDPUseEVPI = True
    SDDPNrEVPIScenario = 1
    SDDPDebugSolveAverage = False
    SDDPMinimumNrIterationBetweenTest = 30
    SDDPNrItNoImproveLBBeforeTest = 10
    SDDPDurationBeforeIncreaseForwardSample = 3600
    SDDPSaveInExcel = False
    SDDPFixSetupStrategy = False
    SDDPFirstForwardWithEVPI = False

    PHIterationLimit = 10000
    PHConvergenceTolerence = 0.0001
    PHMultiplier= 0.1

    MLLSNrIterationBeforeTabu = 9999999
    MLLSTabuList = 5
    MLLSNrIterationTabu = 50
    MLLSPercentFilter  = 50
    MLType = "NN"

    @staticmethod
    def IsDeterministic(s):
        result = s == Constants.Average \
                 or s == Constants.AverageSS \
                 or s == Constants.AverageSSGrave
        return result

    @staticmethod
    def IsSDDPBased(s):
        result = s == Constants.SDDP \
                 or s == Constants.MLLocalSearch \
                 or s == Constants.Hybrid
        return result


    @staticmethod
    def UseSafetyStock(s):
        result = s == Constants.AverageSS \
                 or s == Constants.AverageSSGrave \
                 or s == Constants.AverageSSDyn \
                 or s == Constants.AverageSSStat
        return result

    @staticmethod
    def IsQMCMethos(s):
       result = s in [Constants.QMC, Constants.RQMC]
       return result

    @staticmethod
    def IsRuleWithGrave(s):
        return False

    @staticmethod
    def IsRule(s):
        return False

    @staticmethod
    def GetEvaluationFolder():
        if Constants.PrintSolutionFileInTMP:
            return "/tmp/thesim/Evaluations/"
        else:
            return "./Evaluations/"

    @staticmethod
    def GetPathCPLEXLog():
        if Constants.PrintSolutionFileInTMP:
            return "/tmp/thesim/CPLEXLog/"
        else:
            return "./CPLEXLog/"

