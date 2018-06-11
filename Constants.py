from __future__ import absolute_import, division, print_function
class Constants( object ):
    PathInstances = "./Instances/"
    PathCPLEXLog = "./CPLEXLog/"
    EvaluationFolder = "./Evaluations/"# "/tmp/thesim/Evaluations/"

    #Scenario sampling methods:
    MonteCarlo = "MC"
    QMC = "QMC"
    RQMC = "RQMC"
    All = "all"

    #Method
    MIP = "MIP"
    SDDP = "SDDP"

    #Demand distributions:
    Lumpy = "Lumpy"
    SlowMoving = "SlowMoving"
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

    #Action
    Solve ="Solve"
    Evaluate = "Evaluate"

    #Decision Framework
    RollingHorizon = "RH"
    Fix = "Fix"
    Resolve = "Re-solve"

    #The set of seeds used for random number generator
    SeedArray = [2934, 875, 3545, 765, 546, 768, 242, 375, 142, 236, 788]

    #Running option
    Debug = False
    PrintSolutionFileToExcel = True
    PrintDetailsExcelFiles = False
    PrintOnlyFirstStageDecision = True
    RunEvaluationInSeparatedJob = False
    PrintScenarios = False
    PrintSolutionFileInTMP = False
    LauchEvalAfterSolve = True

    #Code parameter
    Infinity = 9999999999999
    AlgorithmTimeLimit = 600

    #SDDPparameters
    AlgorithmOptimalityTolerence = 0.05
    SDDPIterationLimit = 10000
    SDDPPrintDebugLPFiles = False
    PrintSDDPTrace = True
    GenerateStrongCut = True
    SDDPRunSigleTree = False

    SDDPNrScenarioForwardPass = 1
    SDDPNrScenarioBackwardPass = 1
    SDDPNrScenarioTest = 500

    SolveRelaxationFirst = True
    SDDPNrIterationRelax = 500
    SDDPGapRelax = 0.001

    SDDPUseValidInequalities = False

    SDDPGenerateCutWith2Stage = False

    SDDPCleanCuts = False

    SDDPUseEVPI = True
    SDDPNrEVPIScenario = 1
    SDDPDebugSolveAverage = False

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
