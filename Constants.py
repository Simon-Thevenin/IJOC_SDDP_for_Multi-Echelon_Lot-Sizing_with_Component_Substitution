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
    PrintSolutionFileInTMP = True
    LauchEvalAfterSolve = False

    #Code parameter
    Infinity = 9999999999999
    AlgorithmTimeLimit = 3600
    MIPBasedOnSymetricTree = True

    #SDDPparameters
    AlgorithmOptimalityTolerence = 0.01#0.0005
    SDDPIterationLimit = 100000
    SDDPPrintDebugLPFiles = False
    PrintSDDPTrace = True
    GenerateStrongCut = False
    SDDPRunSigleTree = False

    #SDDPNrScenarioForwardPass = 10
    #SDDPNrScenarioBackwardPass = 10
    SDDPForwardPassInSAATree = True
    SDDPIncreaseNrScenarioTest = 10
    SDDPInitNrScenarioTest = 10

    SolveRelaxationFirst = False
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

    @staticmethod
    def IsDeterministic(s):
        result = s == Constants.Average \
                 or s == Constants.AverageSS \
                 or s == Constants.AverageSSGrave
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

