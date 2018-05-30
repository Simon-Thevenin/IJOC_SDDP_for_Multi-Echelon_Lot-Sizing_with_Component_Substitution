from __future__ import absolute_import, division, print_function
from Constants import Constants
from Solver import Solver
from Evaluator import Evaluator
from TestIdentificator import TestIdentificator
from EvaluatorIdentificator import EvaluatorIdentificator
from Instance import Instance
import os
import argparse
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
                                       args.mipsetting)
    EvaluatorIdentifier = EvaluatorIdentificator(policygeneration,  args.nrevaluation, args.timehorizon, args.allscenario)

def Solve(instance):
    global LastFoundSolution
    solver = Solver(instance, TestIdentifier, mipsetting="", evaluatesol=EvaluateSolution)

    solution = solver.Solve()

    LastFoundSolution = solution

    evaluator = Evaluator(instance, TestIdentifier, EvaluatorIdentifier, solver)
    evaluator.RunEvaluation()
    evaluator.GatherEvaluation()

def Evaluate():
    evaluator = Evaluator(TestIdentifier, EvaluatorIdentifier)


if __name__ == '__main__':

    #instance.ReadFromFile("lpopt_input.dat", Constants.NonStationary)
    #instance.ReadInstanceFromExelFile( "G0044432_NonStationary_b2_fe25_en_rk50_ll0_l20_HFalse_c0" )
    #instance.SaveCompleteInstanceInExelFile()

    try:
        CreateRequiredDir()
        parseArguments()

        if TestIdentifier.MIPSetting == "NoOneTree":
            Constants.SDDPRunSigleTree = False
        if TestIdentifier.MIPSetting == "NoFirstCuts":
            Constants.SDDPNrIterationRelax = 0
            Constants.SDDPGenerateCutWith2Stage = False
            Constants.SolveRelaxationFirst = False
        if TestIdentifier.MIPSetting == "NoValidInequalities":
            Constants.SDDPUseValidInequalities = False
        if TestIdentifier.MIPSetting == "NoStongCut":
            Constants.GenerateStrongCut = False

        instance = Instance()
        #instance.DefineAsSuperSmallIntance()
        #instance.ReadFromFile("K0011525", "NonStationary", "Normal")
        #instance.ReadFromFile("03", "NonStationary", "Normal")
        #instance.SaveCompleteInstanceInExelFile()
        instance.ReadInstanceFromExelFile(TestIdentifier.InstanceName)
        #instance.DrawSupplyChain()
    except KeyError:
        print("This instance does not exist.")

    if Action == Constants.Solve:
        Solve(instance)

    if Action == Constants.Evaluate:
        if TestIdentifier.ScenarioSeed == -1:
            Evaluate()
        else:
            EvaluateSingleSol()
