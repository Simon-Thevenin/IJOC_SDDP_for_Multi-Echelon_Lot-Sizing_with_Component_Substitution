from __future__ import absolute_import, division, print_function
from Constants import Constants
from Solver import Solver
from Evaluator import Evaluator
from TestIdentificator import TestIdentificator
from EvaluatorIdentificator import EvaluatorIdentificator
from Instance import Instance
import os
import argparse
from SDDP import SDDP



import csv
import datetime

#Contain the paramter of the test being run and the evaluation method
TestIdentifier = None
EvaluatorIdentifier = None

#Solve or Evaluate
Action = ""

#The solution to evaluate
EvaluateSolution = None


def CreateRequiredDir():
    requireddir = ["./Test","./CPLEXLog", "./Test/Statistic", "./Test/Bounds", "./Test/SolveInfo", "./Solutions", "./Evaluations", "./Temp", ]
    for dir in requireddir:
        if not os.path.exists(dir):
            os.makedirs(dir)

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("Action", help="Evaluate, Solve, VSS, EVPI", type=str)
    parser.add_argument("Instance", help="Cname of the instance.", type=str)
    parser.add_argument("Model", help="Average/YQFix/YFiz .", type=str)
    parser.add_argument("NrScenario", help="the number of scenario used for optimization", type=str)
    parser.add_argument("ScenarioGeneration", help="MC,/RQMC.", type=str)
    parser.add_argument("-s", "--ScenarioSeed", help="The seed used for scenario generation", type=int, default=-1)
    # Optional arguments
    parser.add_argument("-p", "--policy", help="NearestNeighbor", type=str, default="_")
    parser.add_argument("-n", "--nrevaluation", help="nr scenario used for evaluation.", type=int, default=500)
    parser.add_argument("-m", "--method", help="method used to solve", type=str, default="MIP")
    parser.add_argument("-f", "--fixuntil", help="Use with VSS action, how many periods are fixed", type=int, default=0)
    parser.add_argument("-e", "--evpi", help="if true the evpi model is consdiered",  default=False, action='store_true')
    parser.add_argument("-c", "--mipsetting", help="test a specific mip solver parameter",  default="")
    parser.add_argument("-t", "--timehorizon", help="the time horizon used in shiting window.", type=int, default=1)
    parser.add_argument("-a", "--allscenario", help="generate all possible scenario.", type=int, default=0)
    parser.add_argument("-w", "--nrforward", help="number of scenario in the forward pass of sddp.", type=int, default=0)
    parser.add_argument("-d", "--sddpsetting", help="test a specific sddp parameter", default="")
    parser.add_argument("-y", "--hybridphsetting", help="test a specific hybridph parameter", default="")
    parser.add_argument("-z", "--mllocalsearchsetting", help="test a specific mllocalsearch parameter", default="")


    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    global TestIdentifier
    global EvaluatorIdentifier
    global Action
    Action = args.Action
    policygeneration = args.policy
    FixUntilTime = args.fixuntil
    if args.evpi:
        policygeneration ="EVPI"


    TestIdentifier = TestIdentificator(args.Instance,
                                       args.Model,
                                       args.method,
                                       args.ScenarioGeneration,
                                       args.NrScenario,
                                       Constants.SeedArray[args.ScenarioSeed],
                                       args.evpi,
                                       args.nrforward,
                                       args.mipsetting,
                                       args.sddpsetting,
                                       args.hybridphsetting,
                                       args.mllocalsearchsetting)


    EvaluatorIdentifier = EvaluatorIdentificator(policygeneration,  args.nrevaluation, args.timehorizon, args.allscenario)

def Solve(instance):
    global LastFoundSolution
    solver = Solver(instance, TestIdentifier, mipsetting="", evaluatesol=EvaluateSolution)

    solution = solver.Solve()

    LastFoundSolution = solution
    evaluator = Evaluator(instance, TestIdentifier, EvaluatorIdentifier, solver)
    evaluator.RunEvaluation()
    if Constants.LauchEvalAfterSolve and EvaluatorIdentifier.NrEvaluation>0:
        evaluator.GatherEvaluation()



def Evaluate():
    solver = Solver(instance, TestIdentifier, mipsetting="", evaluatesol=EvaluateSolution)


    evaluator = Evaluator(instance, TestIdentifier, EvaluatorIdentifier, solver)



    evaluator.RunEvaluation()
    evaluator.GatherEvaluation()


def GenerateInstances():
    instancecreated = []
    instance = Instance()
    #instance.DefineAsSuperSmallIntance()
    # instance.ReadFromFile("K0011525", "NonStationary", "Normal")
    #instance.ReadFromFile("10", "Lumpy", "Normal")
    #instance.SaveCompleteInstanceInExelFile()
    #instancecreated = instancecreated + [instance.InstanceName]
    #
    # for sc in ["01", "02", "03"]:#, "04", "05"]:
    #     for addtime in [0,1,2]:
    #         instance.ReadFromFile(sc, "Lumpy", "Normal", additionaltimehorizon=addtime)
    #         instance.SaveCompleteInstanceInExelFile()
    #         instancecreated = instancecreated + [instance.InstanceName]
    # #
    # instance.ReadFromFile("03", "Lumpy", "Normal")
    # instance.SaveCompleteInstanceInExelFile()
    # instancecreated = instancecreated + [instance.InstanceName]
    # #
    # instance.ReadFromFile("04", "Lumpy", "Normal")
    # instance.SaveCompleteInstanceInExelFile()
    # instancecreated = instancecreated + [instance.InstanceName]
    # #
    # instance.ReadFromFile("05", "Lumpy", "Normal")
    # instance.SaveCompleteInstanceInExelFile()
    # instancecreated = instancecreated + [instance.InstanceName]



    normalhorizon = 4
    normalalternate = 4
    normalcostalternate = 0.1
    for name in ["G0041331","G0041111", "K0011311", "K0011331"]:#,"G0041311","K0011111","K0011311","G0041131","G0041331","K0011131","K0011331"]:
        for horizon in [2, 4, 6, 8, 10]:
            instance.ReadFromFile(name, "Lumpy", longtimehoizon=True, longtimehorizonperiod=horizon,
                                  nralternate=normalalternate, costalternate=normalcostalternate)
            instance.SaveCompleteInstanceInExelFile()
            instancecreated = instancecreated + [instance.InstanceName]
            instance.ReadFromFile(name, "Binomial", longtimehoizon=True, longtimehorizonperiod=horizon,
                                  nralternate=normalalternate, costalternate=normalcostalternate)
            instance.SaveCompleteInstanceInExelFile()
            instancecreated = instancecreated + [instance.InstanceName]

        for nralternates in [0,2,4,6]:
            instance.ReadFromFile(name, "Lumpy", longtimehoizon=True, longtimehorizonperiod=normalhorizon,
                                  nralternate=nralternates, costalternate=normalcostalternate)
            instance.SaveCompleteInstanceInExelFile()
            instancecreated = instancecreated + [instance.InstanceName]

        for costalternates in [0,0.1,1]:
            instance.ReadFromFile(name, "Lumpy", longtimehoizon=True, longtimehorizonperiod=normalhorizon,
                                  nralternate=normalalternate, costalternate=costalternates)
            instance.SaveCompleteInstanceInExelFile()
            instancecreated = instancecreated + [instance.InstanceName]


    csvfile = open("./Instances/InstancesToSolve.csv", 'wb')
    data_rwriter = csv.writer(csvfile, delimiter=",", skipinitialspace=True)
    data_rwriter.writerow(instancecreated)

if __name__ == '__main__':

    #instance.ReadFromFile("lpopt_input.dat", Constants.NonStationary)
    #instance.ReadInstanceFromExelFile( "G0044432_NonStationary_b2_fe25_en_rk50_ll0_l20_HFalse_c0" )
    #instance.SaveCompleteInstanceInExelFile()
    # if Constants.UseGUI:
    #     fenetre = Tk()
    #
    #     label = Label(fenetre, text="SCM")
    #     label.pack()
    #
    #     bouton = Button(fenetre, text="Fermer", command=fenetre.quit)
    #     bouton.pack()
    #
    #     fenetre.mainloop()

    try:
        CreateRequiredDir()
        parseArguments()

        if TestIdentifier.MIPSetting == "NoFirstCuts":
            Constants.SDDPGenerateCutWith2Stage = False
            Constants.SolveRelaxationFirst = False
        if TestIdentifier.MIPSetting == "NoEVPI":
            Constants.SDDPUseEVPI = False
        if TestIdentifier.MIPSetting == "NoStongCut":
            Constants.GenerateStrongCut = False
        if TestIdentifier.MIPSetting == "NoSingleTree":
            Constants.SDDPRunSigleTree = False
        if TestIdentifier.MIPSetting == "WithLPTree":
            Constants.SDDPFirstForwardWithEVPI = True
            Constants.GenerateStrongCut = False

        if TestIdentifier.MIPSetting == "WithFixedSetupsNoScenarioTree":
            Constants.SDDPFixSetupStrategy = True
            SDDPRunSigleTree = False

        if TestIdentifier.MIPSetting == "WithFixedSetups":
            Constants.SDDPFixSetupStrategy = True

        if TestIdentifier.MIPSetting == "SymetricMIP":
            Constants.MIPBasedOnSymetricTree = True
            Constants.SDDPForwardPassInSAATree = True

        instance = Instance()
        #instance.DefineAsSuperSmallIntance()
        #instance.DefineAsTwoItemIntance()
        # instance.ReadFromFile("K0011525", "NonStationary", "Normal")

        #GenerateInstances()

        instance.ReadInstanceFromExelFile(TestIdentifier.InstanceName)
        Constants.AlgorithmTimeLimit = 900 *(instance.NrTimeBucket-instance.NrTimeBucketWithoutUncertaintyBefore)
        #instance.DrawSupplyChain()
    except KeyError:
        print(KeyError.message)
        print("This instance does not exist.")

    if Action == Constants.Solve:
        Solve(instance)

    if Action == Constants.Evaluate:
        Evaluate()

