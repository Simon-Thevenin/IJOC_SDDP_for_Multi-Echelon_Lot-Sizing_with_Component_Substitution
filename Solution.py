from __future__ import absolute_import, division, print_function
from datetime import datetime
import math
import csv
from ScenarioTree import ScenarioTree
from Constants import Constants
from Tool import Tool
from Instance import Instance
import openpyxl as opxl
from ast import literal_eval
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

class Solution(object):

    #constructor
    def __init__(self, instance=None, solquantity=None, solproduction=None, solinventory=None, solbackorder=None, solconsumption=None, scenarioset=None, scenriotree=None, partialsolution=False):
        self.Instance = instance
        #The set of scenario on which the solution is found
        self.Scenarioset = scenarioset
        self.ScenarioTree = scenriotree
        if not self.Scenarioset is None:
            self.SenarioNrset = range(len(self.Scenarioset))

        self.ProductionQuantity = solquantity
        self.InventoryLevel = solinventory
        self.Production = solproduction
        self.BackOrder = solbackorder
        self.Consumption = solconsumption
        self.InventoryCost = -1
        self.BackOrderCost = -1
        self.ConsumptionCost = -1
        self.LostsaleCost =-1
        self.VariableCost = -1
        self.InSamplePercentOnTime = -1
        self.SetupCost = -1
        self.TotalCost =-1
        self.IsPartialSolution = partialsolution
        self.NotCompleteSolution = False
        self.IsSDDPSolution = False

        if instance is not None and not self.IsPartialSolution:
            self.ComputeCost()
        #The attribute below compute some statistic on the solution
        self.InSampleAverageInventory = []
        self.InSampleAverageBackOrder = []
        self.InSampleAverageOnTime = []
        self.InSampleAverageQuantity = []
        self.InSampleTotalDemand = -1
        self.InSampleTotalBackOrder = -1
        self.InSampleTotalLostSale = -1
        self.InSampleAverageDemand = -1
        self.InSampleAverageBackOrder = -1
        self.InSampleAverageLostSale = -1

        self.SValue = []
        if not instance is None:
            self.FixedQuantity = [[-1 for p in instance.ProductSet] for t in instance.TimeBucketSet]
        else:
            self.FixedQuantity = []
        # The objecie value as outputed by CPLEx,
        self.CplexCost = -1
        self.CplexGap = -1
        self.CplexTime = 0
        self.TotalTime = 0
        self.CplexNrConstraints = -1
        self.CplexNrVariables = -1
        self.MLLocalSearchLB = -1
        self.MLLocalSearchTimeBestSol = -1
        self.PHCost = -1
        self.PHNrIteration = -1

        self.SDDPLB = -1
        self.SDDPExpUB = -1
        self.SDDPNrIteration = -1

        self.LocalSearchIteration = -1

        self.SDDPTimeBackward = -1
        self.SDDPTimeForwardNoTest = -1
        self.SDDPTimeForwardTest = -1


    #return the path to file where excel the solution is saved
    def GetSolutionFileName(self, description):
        if Constants.PrintSolutionFileInTMP:
            result = "/tmp/thesim/Solutions/" + description + "_Solution.xlsx"
        else:
            result ="./Solutions/"+  description + "_Solution.xlsx"
        return result

    # return the path to binary file of the solution is saved
    def GetSolutionPickleFileNameStart(self, description, dataframename):
        if Constants.PrintSolutionFileInTMP:
            result = "/tmp/thesim/Solutions/" + description + "_" + dataframename
        else:
            result ="./Solutions/"+  description + "_" + dataframename
        return result

    #Cerate a dataframe with the general information about the solution
    def GetGeneralInfoDf(self):
        model = ""
        if not self.ScenarioTree is None and not self.ScenarioTree.Owner is None:
            model = self.ScenarioTree.Owner.Model
        else:
            model = "Rule"
        general = [self.Instance.InstanceName, self.Instance.Distribution, model,
                   self.CplexCost, self.CplexTime, self.TotalTime, self.CplexGap, self.CplexNrConstraints,
                   self.CplexNrVariables, self.SDDPLB, self.SDDPExpUB, self.SDDPNrIteration,  self.SDDPTimeBackward,
                   self.SDDPTimeForwardNoTest, self.SDDPTimeForwardTest, self.MLLocalSearchLB, self.MLLocalSearchTimeBestSol, self.LocalSearchIteration, self.PHCost,
                   self.PHNrIteration, self.IsPartialSolution, self.IsSDDPSolution]

        
        columnstab = ["Name", "Distribution", "Model", "CplexCost", "CplexTime", "TotalTime", "CplexGap", "CplexNrConstraints",
                      "CplexNrVariables", "SDDP_LB", "SDDP_ExpUB", "SDDP_NrIteration",  "SDDPTimeBackward",
                   "SDDPTimeForwardNoTest", "SDDPTimeForwardTest", "MLLocalSearchLB", "MLLocalSearchTimeBestSol", "LocalSearchIterations", "PH_Cost", "PH_NrIteration", "IsPartialSolution", "ISSDDPSolution"]
        generaldf = pd.DataFrame(general, index=columnstab)
        return generaldf

    # This function print the solution different pickle files
    def PrintToPickle(self, description):
            prodquantitydf, inventorydf, productiondf, bbackorderdf, consumptiondf, fixedqvaluesdf = self.DataFrameFromList()

            prodquantitydf.to_pickle(self.GetSolutionPickleFileNameStart(description, 'ProductionQuantity') )
            productiondf.to_pickle(self.GetSolutionPickleFileNameStart(description,  'Production'))
            inventorydf.to_pickle(self.GetSolutionPickleFileNameStart(description,  'InventoryLevel'))
            bbackorderdf.to_pickle(self.GetSolutionPickleFileNameStart(description,  'BackOrder'))
            consumptiondf.to_pickle(self.GetSolutionPickleFileNameStart(description, 'Consumption'))
            #svaluedf.to_pickle(self.GetSolutionPickleFileNameStart(description,  'SValue'))
            fixedqvaluesdf.to_pickle(self.GetSolutionPickleFileNameStart(description,  'FixedQvalue'))

            generaldf = self.GetGeneralInfoDf()
            generaldf.to_pickle(self.GetSolutionPickleFileNameStart(description, "Generic"))

            scenariotreeinfo = [self.Instance.InstanceName, self.ScenarioTree.Seed, self.ScenarioTree.TreeStructure,
                                self.ScenarioTree.AverageScenarioTree, self.ScenarioTree.ScenarioGenerationMethod]
            columnstab = ["Name", "Seed", "TreeStructure", "AverageScenarioTree", "ScenarioGenerationMethod"]
            scenariotreeinfo = pd.DataFrame(scenariotreeinfo, index=columnstab)
            scenariotreeinfo.to_pickle( self.GetSolutionPickleFileNameStart(description,  "ScenarioTree") )

    #This function print the solution in an Excel file in the folde "Solutions"
    def PrintToExcel(self, description):
        prodquantitydf, inventorydf, productiondf, bbackorderdf, consumptiondf,  fixedqvaluesdf = self.DataFrameFromList()
        writer = pd.ExcelWriter(self.GetSolutionFileName(description), engine='openpyxl')
        prodquantitydf.to_excel(writer, 'ProductionQuantity')
        productiondf.to_excel(writer, 'Production')
        inventorydf.to_excel(writer, 'InventoryLevel')
        bbackorderdf.to_excel(writer, 'BackOrder')
        consumptiondf.to_excel(writer, 'Consumption')
        #svaluedf.to_excel(writer, 'SValue')
        fixedqvaluesdf.to_excel(writer, 'FixedQvalue')

        generaldf = self.GetGeneralInfoDf()
        generaldf.to_excel(writer, "Generic")

        scenariotreeinfo = [self.Instance.InstanceName, self.ScenarioTree.Seed, self.ScenarioTree.TreeStructure, self.ScenarioTree.AverageScenarioTree, self.ScenarioTree.ScenarioGenerationMethod]
        columnstab = ["Name", "Seed", "TreeStructure", "AverageScenarioTree", "ScenarioGenerationMethod" ]
        scenariotreeinfo = pd.DataFrame(scenariotreeinfo, index=columnstab)
        scenariotreeinfo.to_excel(writer, "ScenarioTree")



        writer.save()

    #This function return a set of dataframes describing the content of the excel file
    def ReadExcelFiles(self, description, index="", indexbackorder=""):
        # The supplychain is defined in the sheet named "01_LL" and the data are in the sheet "01_SD"
        prodquantitydf = Tool.ReadMultiIndexDataFrame(self.GetSolutionFileName(description), "ProductionQuantity")
        productiondf = Tool.ReadMultiIndexDataFrame(self.GetSolutionFileName(description), "Production")
        inventorydf = Tool.ReadMultiIndexDataFrame(self.GetSolutionFileName(description), "InventoryLevel")
        bbackorderdf = Tool.ReadMultiIndexDataFrame(self.GetSolutionFileName(description), "BackOrder")
        consumptiondf = Tool.ReadMultiIndexDataFrame(self.GetSolutionFileName(description), "Consumption")


        wb2 = opxl.load_workbook(self.GetSolutionFileName(description))
       # svaluedf = Tool.ReadDataFrame(wb2, 'SValue')
        fixedqvaluesdf = Tool.ReadDataFrame(wb2, 'FixedQvalue')
        instanceinfo = Tool.ReadDataFrame(wb2, "Generic")
        scenariotreeinfo = Tool.ReadDataFrame(wb2, "ScenarioTree")

        prodquantitydf.index = index
        productiondf.index = index
        inventorydf.index = index
        bbackorderdf.index = [index[p] for p in indexbackorder]
        return prodquantitydf, productiondf, inventorydf, bbackorderdf, consumptiondf,  fixedqvaluesdf, instanceinfo, scenariotreeinfo

    #This function return a set of dataframes describing the content of the binary file
    def ReadPickleFiles(self, description):
        # The supplychain is defined in the sheet named "01_LL" and the data are in the sheet "01_SD"
        prodquantitydf = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'ProductionQuantity'))
        productiondf = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'Production'))
        inventorydf = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'InventoryLevel'))
        bbackorderdf = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'BackOrder'))
        consumptiondf = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'Consumption'))
        #svaluedf = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'SValue'))
        fixedqvaluesdf = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, 'FixedQvalue'))
        instanceinfo = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, "Generic"))
        scenariotreeinfo = pd.read_pickle(self.GetSolutionPickleFileNameStart(description, "ScenarioTree"))

        return prodquantitydf, productiondf, inventorydf, bbackorderdf, consumptiondf, fixedqvaluesdf, instanceinfo, scenariotreeinfo


    #This function set the attributes of the solution from the excel/binary file
    def ReadFromFile(self, description):
        if Constants.PrintSolutionFileToExcel:
            wb2 = opxl.load_workbook(self.GetSolutionFileName(description))
            instanceinfo = Tool.ReadDataFrame(wb2, "Generic")
            self.Instance = Instance()
            self.Instance.ReadInstanceFromExelFile(instanceinfo.at['Name', 0])
            prodquantitydf, productiondf, inventorydf, bbackorderdf, consumptiondf,  fixedqvaluesdf, instanceinfo, \
            scenariotreeinfo = self.ReadExcelFiles(description, index=self.Instance.ProductName, indexbackorder=self.Instance.ProductWithExternalDemand)
        else:
            prodquantitydf, productiondf, inventorydf, bbackorderdf, consumptiondf,  fixedqvaluesdf, instanceinfo, \
            scenariotreeinfo = self.ReadPickleFiles(description)



        self.Instance = Instance()
        if Constants.Debug:
            print("Load instance:%r" % instanceinfo.at['Name', 0])
        self.Instance.ReadInstanceFromExelFile(instanceinfo.at['Name', 0])


        scenariogenerationm = scenariotreeinfo.at['ScenarioGenerationMethod', 0]
        avgscenariotree = scenariotreeinfo.at['AverageScenarioTree', 0]
        scenariotreeseed = int(scenariotreeinfo.at['Seed', 0])
        branchingstructure  = literal_eval(str(scenariotreeinfo.at['TreeStructure', 0]))
        model = instanceinfo.at['Model', 0]
        RQMCForYQfix = (model == Constants.ModelYQFix and (Constants.IsQMCMethos(scenariogenerationm)))

        self.ScenarioTree = ScenarioTree(instance=self.Instance,
                                         branchperlevel=branchingstructure,
                                         seed=scenariotreeseed,
                                         averagescenariotree=avgscenariotree,
                                         scenariogenerationmethod=scenariogenerationm,
                                         model=model)

        self.IsPartialSolution = instanceinfo.at['IsPartialSolution', 0]
        self.IsSDDPSolution = instanceinfo.at['ISSDDPSolution', 0]
        self.CplexCost = instanceinfo.at['CplexCost', 0]
        self.CplexTime = instanceinfo.at['CplexTime', 0]
        self.PHCost = instanceinfo.at['PH_Cost', 0]
        self.PHNrIteration = instanceinfo.at['PH_NrIteration', 0]
        self.TotalTime = instanceinfo.at['TotalTime', 0]
        self.CplexGap = instanceinfo.at['CplexGap', 0]
        self.CplexNrConstraints = instanceinfo.at['CplexNrConstraints', 0]
        self.CplexNrVariables = instanceinfo.at['CplexNrVariables', 0]

        self.SDDPLB = instanceinfo.at['SDDP_LB', 0]
        self.SDDPExpUB = instanceinfo.at['SDDP_ExpUB', 0]
        self.SDDPNrIteration = instanceinfo.at['SDDP_NrIteration', 0]

        self.LocalSearchIteration = instanceinfo.at['LocalSearchIterations', 0]
        self.MLLocalSearchLB = instanceinfo.at['MLLocalSearchLB', 0]
        self.MLLocalSearchTimeBestSol = instanceinfo.at['MLLocalSearchTimeBestSol', 0]


        self.SDDPTimeBackward = instanceinfo.at['SDDPTimeBackward', 0]
        self.SDDPTimeForwardNoTest = instanceinfo.at['SDDPTimeForwardNoTest', 0]
        self.SDDPTimeForwardTest = instanceinfo.at['SDDPTimeForwardTest', 0]
        
        self.Scenarioset = self.ScenarioTree.GetAllScenarios(False)
        if self.IsPartialSolution:
            self.Scenarioset = [self.Scenarioset[0]]
        self.SenarioNrset = range(len(self.Scenarioset))
        self.ListFromDataFrame(prodquantitydf, inventorydf, productiondf, bbackorderdf, consumptiondf, fixedqvaluesdf)
        if not self.IsPartialSolution:
            self.ComputeCost()

            if model <> Constants.ModelYQFix:
                self.ScenarioTree.FillQuantityToOrderFromMRPSolution(self)

    #This function prints a solution
    def Print(self):
        prodquantitydf, inventorydf, productiondf, bbackorderdf, consumptiondf, fixedqvaluesdf = self.DataFrameFromList()
        print("production ( cost: %r): \n %r" % ( self.SetupCost, productiondf))
        print("production quantities ( cost: %r): \n %r" % ( self.VariableCost, prodquantitydf) )
        print("inventory levels at the end of the periods: ( cost: %r ) \n %r" % (self.InventoryCost, inventorydf))
        print("backorder quantities:  ( cost: %r ) \n %r" % (self.BackOrderCost, bbackorderdf))
        print("Consumption: ( cost: %r ) \n %r " %(self.ConsumptionCost, consumptiondf))
       # print("S values: \n %r" % svaluedf)
        print("Fixed Q values: \n %r" % fixedqvaluesdf)

    #This funciton conpute the different costs (inventory, backorder, setups) associated with the solution.
    def ComputeCost(self):
        self.TotalCost, self.InventoryCost, self.BackOrderCost, self.SetupCost, \
        self.LostsaleCost, self.VariableCost, self.ConsumptionCost = self.GetCostInInterval( self.Instance.TimeBucketSet )

    #This function return the costs encountered in a specific time interval
    def GetCostInInterval(self, timerange):
        inventorycost = 0
        backordercost = 0
        setupcost = 0
        lostsalecost = 0
        variablecost = 0
        consumptioncost = 0
        gammas = [math.pow(self.Instance.Gamma, t) for t in self.Instance.TimeBucketSet]

        for w in range(len(self.Scenarioset)):
            for t in timerange:
                for p in self.Instance.ProductSet:

                    inventorycost += self.InventoryLevel[w][t][p] \
                                          * self.Instance.InventoryCosts[p] \
                                          * gammas[t] \
                                          * self.Scenarioset[w].Probability

                    setupcost += self.Production[w][t][p] \
                                      * self.Instance.SetupCosts[p] \
                                      * gammas[t] \
                                      * self.Scenarioset[w].Probability

                    variablecost += self.ProductionQuantity[w][t][p] \
                                          * self.Instance.VariableCost[p] \
                                          * gammas[t] \
                                          * self.Scenarioset[w].Probability

                    if self.Instance.HasExternalDemand[p]:
                        if t < self.Instance.NrTimeBucket - 1:
                            backordercost += self.BackOrder[w][t][
                                                      self.Instance.ProductWithExternalDemandIndex[p]] \
                                                  * self.Instance.BackorderCosts[p] \
                                                  * gammas[t] \
                                                  * self.Scenarioset[w].Probability
                        else:
                            lostsalecost += self.BackOrder[w][t][self.Instance.ProductWithExternalDemandIndex[p]] \
                                                 * self.Instance.LostSaleCost[p] \
                                                 * gammas[t] \
                                                 * self.Scenarioset[w].Probability

                    for q in self.Instance.ProductSet:
                        if self.Instance.Alternates[p][q] \
                                and self.Consumption[w][t][p][q] > 0:

                            consumptioncost += self.Consumption[w][t][p][q] \
                                                * self.Instance.GetComsumptionCost(p,q)\
                                                *gammas[t] \
                                                * self.Scenarioset[w].Probability
                totalcost = inventorycost + backordercost + setupcost + lostsalecost + variablecost + consumptioncost
        return totalcost, inventorycost, backordercost, setupcost, lostsalecost, variablecost, consumptioncost

    #This function return the number of time bucket covered by the solution
    def GetConsideredTimeBucket(self):
        result = self.Instance.TimeBucketSet
        if self.IsPartialSolution:
            result = range(self.Instance.NrTimeBucketWithoutUncertaintyBefore +1)
        if self.IsSDDPSolution:
            result = [0]
        return result

    #This function return the number of scenario for which the solution has values
    def GetConsideredScenarioset(self):
        result = self.Scenarioset
        if self.IsPartialSolution:
            result = [0]
        return result

    #This function return a set of dataframes descirbing the solution
    def DataFrameFromList(self):
        scenarioset = range(len(self.Scenarioset))
        timebucketset = self.GetConsideredTimeBucket()
        solquantity = [[self.ProductionQuantity[s][t][p] for t in timebucketset for s in scenarioset]
                        for p in self.Instance.ProductSet]
        solinventory = [[self.InventoryLevel[s][t][p] for t in timebucketset for s in scenarioset]
                        for p in self.Instance.ProductSet]
        solproduction = [[self.Production[s][t][p] for t in self.Instance.TimeBucketSet for s in scenarioset]
                         for p in self.Instance.ProductSet]
        solbackorder = [[self.BackOrder[s][t][self.Instance.ProductWithExternalDemandIndex[p]]
                         for t in timebucketset for s in scenarioset] for p in self.Instance.ProductWithExternalDemand]
       # svalue = [[self.SValue[t][p] for t in timebucketset] for p in self.Instance.ProductSet]
        fixedqvalues = [[self.FixedQuantity[t][p] for t in timebucketset] for p in self.Instance.ProductSet]

        iterables = [timebucketset, range(len(self.Scenarioset))]
        multiindex = pd.MultiIndex.from_product(iterables, names=['time', 'scenario'])
        prodquantitydf = pd.DataFrame(solquantity, index=self.Instance.ProductName, columns=multiindex)
        prodquantitydf.index.name = "Product"
        inventorydf = pd.DataFrame(solinventory, index=self.Instance.ProductName, columns=multiindex)
        inventorydf.index.name = "Product"

        solconsumption = [[self.Consumption[s][t][c[0]][c[1]]
                           for t in timebucketset for s in scenarioset]
                          for c in self.Instance.ConsumptionSet]

        consumptionname = [c[2] for c in self.Instance.ConsumptionSet]
        consumptiondf = pd.DataFrame(solconsumption, index=consumptionname, columns=multiindex)
        consumptiondf.index.name = "Product"

        #Production variables are decided at stage 1 for the complete horizon
        iterablesproduction = [range(len(self.Instance.TimeBucketSet)), range(len(self.Scenarioset))]
        multiindexproduction = pd.MultiIndex.from_product(iterablesproduction, names=['time', 'scenario'])
        productiondf = pd.DataFrame(solproduction, index=self.Instance.ProductName, columns=multiindexproduction)
        productiondf.index.name = "Product"
        nameproductwithextternaldemand = [self.Instance.ProductName[p] for p in self.Instance.ProductWithExternalDemand]
        bbackorderdf = pd.DataFrame(solbackorder, index=nameproductwithextternaldemand, columns=multiindex)
        bbackorderdf.index.name = "Product"
        #svaluedf = pd.DataFrame(svalue, index=self.Instance.ProductName, columns=timebucketset)
        fixedqvaluedf = pd.DataFrame(fixedqvalues, index=self.Instance.ProductName, columns=timebucketset)

        return prodquantitydf, inventorydf, productiondf, bbackorderdf, consumptiondf, fixedqvaluedf

    #This function creates a solution from the set of dataframe given in paramter
    def ListFromDataFrame(self, prodquantitydf, inventorydf, productiondf, bbackorderdf, consumptiondf, fixedqvaluedf):
        scenarioset = range(len(self.Scenarioset))
        timebucketset = self.GetConsideredTimeBucket()
        self.ProductionQuantity = [[[prodquantitydf.loc[str(self.Instance.ProductName[p]),(t,s)]
                                     for p in self.Instance.ProductSet] for t in timebucketset] for s in scenarioset]
        self.InventoryLevel = [[[inventorydf.loc[self.Instance.ProductName[p], (t, s)]
                                 for p in self.Instance.ProductSet] for t in timebucketset] for s in scenarioset ]
        self.Production = [[[productiondf.loc[self.Instance.ProductName[p], (t, s)] for p in self.Instance.ProductSet]
                            for t in self.Instance.TimeBucketSet] for s in scenarioset]
        self.BackOrder = [[[bbackorderdf.loc[self.Instance.ProductName[p], (t, s)]
                            for p in self.Instance.ProductWithExternalDemand] for t in timebucketset] for s in scenarioset]
        self.Consumption = [[[bbackorderdf.loc[self.Instance.ProductName[p], (t, s)]
                            for p in self.Instance.ProductWithExternalDemand] for t in timebucketset] for s in
                          scenarioset]

        self.Consumption = [[[[-1 for p in self.Instance.ProductSet]
                            for q in self.Instance.ProductSet]
                           for t in timebucketset]
                          for w in scenarioset]
        for c in self.Instance.ConsumptionSet:
            for t in timebucketset:
                for w in scenarioset:
                    self.Consumption[w][t][c[0]][c[1]] = consumptiondf.loc[c[2], (t, w)]


        #self.SValue = [[svaluedf.loc[ self.Instance.ProductName[p], t]
        #                    for p in self.Instance.ProductSet]
        #                    for t in timebucketset]
        self.FixedQuantity = [[fixedqvaluedf.loc[self.Instance.ProductName[p], t]
                            for p in self.Instance.ProductSet]
                            for t in timebucketset]


    def DeleteNonFirstStageDecision(self):
        timebucketset = range(self.Instance.NrTimeBucketWithoutUncertaintyBefore + 1)
        self.IsPartialSolution = True
        self.Scenarioset = [None]
        self.SenarioNrset = [0]
        self.ProductionQuantity = [[[self.ProductionQuantity[w][t][p] for p in self.Instance.ProductSet]
                                    for t in timebucketset] for w in self.SenarioNrset]
        self.InventoryLevel = [[[self.InventoryLevel[w][t][p] for p in self.Instance.ProductSet]
                                for t in timebucketset] for w in self.SenarioNrset]
        self.Production = [[[self.Production[w][t][p] for p in self.Instance.ProductSet]
                            for t in self.Instance.TimeBucketSet] for w in self.SenarioNrset]
        self.BackOrder = [[[self.BackOrder[w][t][self.Instance.ProductWithExternalDemandIndex[p]] for p in self.Instance.ProductWithExternalDemand]
                           for t in timebucketset] for w in self.SenarioNrset]

    #This function compute some statistic on the current solution
    def ComputeStatistics(self):

        self.InSampleAverageInventory = [[sum(self.InventoryLevel[w][t][p] for w in self.SenarioNrset)
                                          /len(self.SenarioNrset)
                                           for p in self.Instance.ProductSet]
                                           for t in self.Instance.TimeBucketSet]

        self.InSampleAverageBackOrder = [[sum( self.BackOrder[w][t][ self.Instance.ProductWithExternalDemandIndex[p] ]
                                               for w in self.SenarioNrset)
                                          /len(self.SenarioNrset )
                                              for p in self.Instance.ProductWithExternalDemand]
                                                for t in self.Instance.TimeBucketSet]

        self.InSampleAverageConsumption = [[sum(self.Consumption[w][t][p][q] for w in self.SenarioNrset for t in self.Instance.TimeBucketSet)
                                         / (len(self.SenarioNrset) * len(self.Instance.TimeBucketSet))
                                         for p in self.Instance.ProductSet]
                                         for q in self.Instance.ProductSet ]

        self.InSampleAverageQuantity = [[sum( self.ProductionQuantity[w][t][p] for w in self.SenarioNrset)
                                         /len(self.SenarioNrset)
                                            for p in self.Instance.ProductSet]
                                         for t in self.Instance.TimeBucketSet]

        self.InSampleAverageSetup = [[sum(self.Production[w][t][p] for w in self.SenarioNrset)
                                      /len(self.SenarioNrset)
                                         for p in self.Instance.ProductSet]
                                       for t in self.Instance.TimeBucketSet]

        self.InSampleAverageOnTime = [[(sum(max([self.Scenarioset[s].Demands[t][p]
                                                 - self.BackOrder[s][t][self.Instance.ProductWithExternalDemandIndex[p]],0])
                                           for s in self.SenarioNrset)
                                             / len(self.SenarioNrset))
                                        for p in self.Instance.ProductWithExternalDemand]
                                        for t in self.Instance.TimeBucketSet]

        self.InSampleTotalDemandPerScenario = [sum(sum(s.Demands[t][p] for p in self.Instance.ProductSet)
                                                   for t in self.Instance.TimeBucketSet)
                                               for s in self.Scenarioset]

        totaldemand = sum(self.InSampleTotalDemandPerScenario)

        backordertime = range( self.Instance.NrTimeBucket - 1)

        self.InSampleTotalOnTimePerScenario = [sum(sum(max([self.Scenarioset[s].Demands[t][p]
                                                             - self.BackOrder[s][t][self.Instance.ProductWithExternalDemandIndex[p]],0])
                                                        for p in self.Instance.ProductWithExternalDemand)
                                                    for t in self.Instance.TimeBucketSet)
                                                for s in self.SenarioNrset]

        self.InSampleTotalBackOrderPerScenario = [sum(self.BackOrder[w][t][self.Instance.ProductWithExternalDemandIndex[p]]
                                                      for t in backordertime
                                                      for p in self.Instance.ProductWithExternalDemand)
                                                  for w in  self.SenarioNrset]

        self.InSampleTotalLostSalePerScenario = [sum(self.BackOrder[w][self.Instance.NrTimeBucket -1][self.Instance.ProductWithExternalDemandIndex[p] ]
                                                     for p in self.Instance.ProductWithExternalDemand) for w in self.SenarioNrset]

        nrscenario = len(self.Scenarioset)

        self.InSampleAverageDemand = sum(self.InSampleTotalDemandPerScenario[s] for s in self.SenarioNrset)/nrscenario
        self.InSamplePercentOnTime = 100 * (sum(self.InSampleTotalOnTimePerScenario[s] for s in self.SenarioNrset))/totaldemand


    #This function print detailed statistics about the obtained solution (avoid using it as it consume memory)
    def PrintDetailExcelStatistic(self, filepostscript, offsetseed, nrevaluation,  testidentifier, evaluationmethod):

        scenarioset = range(len(self.Scenarioset))

        d = datetime.now()
        date = d.strftime('%m_%d_%Y_%H_%M_%S')
        writer = pd.ExcelWriter(
            "./Solutions/" + self.Instance.InstanceName + "_Statistics_" + filepostscript + "_" + date + ".xlsx",
            engine='openpyxl')

        avginventorydf = pd.DataFrame(self.InSampleAverageInventory,
                                      columns=self.Instance.ProductName,
                                      index=self.Instance.TimeBucketSet)

        avginventorydf.to_excel(writer, "AverageInventory")

        avgbackorderdf = pd.DataFrame(self.InSampleAverageBackOrder,
                                      columns=[self.Instance.ProductName[p] for p in
                                               self.Instance.ProductWithExternalDemand],
                                      index=self.Instance.TimeBucketSet)

        avgbackorderdf.to_excel(writer, "AverageBackOrder")

        avgQuantitydf = pd.DataFrame(self.InSampleAverageQuantity,
                                     columns=self.Instance.ProductName,
                                     index=self.Instance.TimeBucketSet)

        avgQuantitydf.to_excel(writer, "AverageQuantity")

        avgSetupdf = pd.DataFrame(self.InSampleAverageSetup,
                                  columns=self.Instance.ProductName,
                                  index=self.Instance.TimeBucketSet)

        avgSetupdf.to_excel(writer, "AverageSetup")

        avgConsumptiondf = pd.DataFrame(self.InSampleAverageConsumption,
                                  columns=self.Instance.ProductName,
                                  index=self.Instance.ProductName)

        avgConsumptiondf.to_excel(writer, "AverageConsumption")


        perscenariodf = pd.DataFrame([self.InSampleTotalDemandPerScenario, self.InSampleTotalBackOrderPerScenario,
                                      self.InSampleTotalLostSalePerScenario],
                                     index=["Total Demand", "Total Backorder", "Total Lost Sales"],
                                     columns=scenarioset)

        perscenariodf.to_excel(writer, "Info Per scenario")

        general = testidentifier.GetAsStringList() + [self.InSampleAverageDemand, offsetseed, nrevaluation, testidentifier.ScenarioSeed, evaluationmethod]
        columnstab = ["Instance", "Model", "Method", "ScenarioGeneration", "NrScenario", "ScenarioSeed",
                      "EVPI", "NrForwardScenario", "mipsetting", "SDDPSetting", "HybridPHSetting", "MLLocalSearchSetting",
                      "Average demand", "offsetseed", "nrevaluation", "solutionseed", "evaluationmethod"]

        generaldf = pd.DataFrame(general, index=columnstab)
        generaldf.to_excel(writer, "General")
        writer.save()

    # Compute the back order per period, and also how long the demand has been backordered
    def GetStatOnBackOrder(self):
        demandofstagetstillbackorder = [[[[0 for p in self.Instance.ProductWithExternalDemand]
                                          for _ in range(currentperiod + 1)]
                                         for currentperiod in self.Instance.TimeBucketSet]
                                        for _ in self.Scenarioset]


        # The portion $\tilde{B}_{p,t}^{n,\omega}$ of the demand due $n$ time periods ago, which is still back-ordered at time $t$ is computed as:
        # \tilde{B}_{p,t}^{n,\omega} = Max(\tilde{B}_{p,t-1}^{n-1,\omega}, B_{p,t}^{\omega} - \tilde{B}_{p,t}^{n-1,\omega})
        for s in self.SenarioNrset:
            for p in self.Instance.ProductWithExternalDemand:
                for currentperiod in self.Instance.TimeBucketSet:
                    for nrperiodago in range(currentperiod + 1):
                        indexp = self.Instance.ProductWithExternalDemandIndex[p]
                        if nrperiodago == 0:
                            demandprevinprev = self.Scenarioset[s].Demands[currentperiod][p]
                        elif currentperiod == 0:
                            demandprevinprev = 0
                        else:
                            demandprevinprev = demandofstagetstillbackorder[s][currentperiod - 1][nrperiodago - 1][indexp]

                        if nrperiodago == 0:
                            demandprevincurrent = 0
                        else:
                            demandprevincurrent = demandofstagetstillbackorder[s][currentperiod][nrperiodago == 0 - 1][ indexp]

                        demandofstagetstillbackorder[s][currentperiod][nrperiodago][indexp] = min(demandprevinprev, max(self.BackOrder[s][currentperiod][ indexp] - demandprevincurrent, 0))

        #The lostsales $\bar{L}_{p,t}^{\omega}$ among the demand due at time $t$ is $\tilde{B}_{p,T}^{n,\omega}$.
        lostsaleamongdemandofstage = [[[ demandofstagetstillbackorder[s][self.Instance.NrTimeBucket -1][nrperiodago ][self.Instance.ProductWithExternalDemandIndex[p] ]
                                             for p in self.Instance.ProductWithExternalDemand]
                                           for nrperiodago in range( self.Instance.NrTimeBucket)]
                                          for s in self.SenarioNrset]

        #The quantity $\bar{B}_{p,t}^{n,\omega}$  of demand of stage $t$ which is backordered during n periods can be computed by:
        #\bar{B}_{p,t}^{n,\omega} =\bar{B}_{p,t + n}^{n,\omega} -\bar{B}_{p,t+ n+ 1}^{n+1,\omega}
        portionbackoredduringtime = [[[[ demandofstagetstillbackorder[s][currentperiod + nrperiod][nrperiod][self.Instance.ProductWithExternalDemandIndex[p]] \
                                           - demandofstagetstillbackorder[s][currentperiod + nrperiod +1][nrperiod +1][self.Instance.ProductWithExternalDemandIndex[p]]
                                             if currentperiod + nrperiod + 1 < self.Instance.NrTimeBucket
                                             else demandofstagetstillbackorder[s][currentperiod + nrperiod][nrperiod][self.Instance.ProductWithExternalDemandIndex[p]]
                                           for p in self.Instance.ProductWithExternalDemand]
                                          for nrperiod in range( self.Instance.NrTimeBucket - currentperiod )]
                                         for currentperiod in self.Instance.TimeBucketSet]
                                        for s in self.SenarioNrset]

        return lostsaleamongdemandofstage, portionbackoredduringtime

    #This function print the statistic in an Excel file
    def PrintStatistics(self, testidentifier, filepostscript, offsetseed, nrevaluation,  evaluationduration, insample, evaluationmethod):

        inventorycoststochasticperiod = -1
        setupcoststochasticperiod = -1
        backordercoststochasticperiod =-1

        # Initialize the average inventory level at each level of the supply chain
        AverageStockAtLevel = [-1 for l in range(self.Instance.NrLevel)]
        nrbackorerxperiod = [ - 1 for t in self.Instance.TimeBucketSet]

        nrlostsale = -1
        #To compute every statistic Constants.PrintOnlyFirstStageDecision should be False
        if (not Constants.PrintOnlyFirstStageDecision) or (not insample):

            avginventorydf = pd.DataFrame(self.InSampleAverageInventory,
                                          columns=self.Instance.ProductName,
                                          index=self.Instance.TimeBucketSet)

            if Constants.PrintDetailsExcelFiles:
                self.PrintDetailExcelStatistic( filepostscript, offsetseed, nrevaluation,  testidentifier, evaluationmethod )

            #Compute the average inventory level at each level of the supply chain
            AverageStockAtLevel = [ ( sum( sum ( avginventorydf.loc[t,self.Instance.ProductName[p]]
                                        for t in self.Instance.TimeBucketSet )
                                            for p in self.Instance.ProductSet if self.Instance.Level[p] == l +1 ) )
                                    / ( sum( 1 for p in self.Instance.ProductSet if self.Instance.Level[p] == l +1 )
                                        * self.Instance.NrTimeBucket )
                                    for l in range( self.Instance.NrLevel ) ]

            lostsaleamongdemandofstage, portionbackoredduringtime = self.GetStatOnBackOrder()

            #Avergae on the senario, product, period
            totaldemand = sum( self.Scenarioset[s].Demands[t][p]
                               for s in self.SenarioNrset
                               for p in self.Instance.ProductWithExternalDemand
                               for t in self.Instance.TimeBucketSet )

            nrbackorerxperiod = [  100 * ( sum( portionbackoredduringtime[s][currentperiod][t][self.Instance.ProductWithExternalDemandIndex[p]]
                                                for p in self.Instance.ProductWithExternalDemand
                                                     for currentperiod in range(self.Instance.NrTimeBucket)
                                                         for s in self.SenarioNrset
                                                            if (t < self.Instance.NrTimeBucket -1 - currentperiod))\
                                                 / totaldemand )
                                                 for t in self.Instance.TimeBucketSet]

            nrlostsale = 100 * sum( lostsaleamongdemandofstage[s][currentperiod][self.Instance.ProductWithExternalDemandIndex[p] ]
                                        for p in self.Instance.ProductWithExternalDemand
                                          for currentperiod in self.Instance.TimeBucketSet
                                            for s in self.SenarioNrset) \
                                / totaldemand

            self.ComputeCost()
            stochasticperiod = range(self.Instance.NrTimeBucketWithoutUncertaintyBefore, self.Instance.NrTimeBucket - self.Instance.NrTimeBucketWithoutUncertaintyAfter)
            totalcoststochasticperiod, \
            inventorycoststochasticperiod, \
            backordercoststochasticperiod, \
            setupcoststochasticperiod,\
            lostsalecoststochasticperiod, \
            variablecoststochasticperiod, \
            consumptioncoststochasticperiod= self.GetCostInInterval( stochasticperiod )
        nrsetups = self.GetNrSetup()
#       averagecoverage = self.GetAverageCoverage()


        kpistat = [ self.CplexCost,
                    self.CplexTime,
                    self.CplexGap,
                    self.CplexNrConstraints,
                    self.CplexNrVariables,
                    self.SDDPLB,
                    self.SDDPExpUB,
                    self.SDDPNrIteration,
                    self.SDDPTimeBackward,
                    self.SDDPTimeForwardNoTest,
                    self.SDDPTimeForwardTest,
                    self.MLLocalSearchLB,
                    self.MLLocalSearchTimeBestSol,
                    self.LocalSearchIteration,
                    self.PHCost,
                    self.PHNrIteration,
                    self.TotalTime,
                    self.SetupCost,
                    self.InventoryCost,
                    self.InSamplePercentOnTime,
                    self.BackOrderCost,
                    self.LostsaleCost,
                    self.VariableCost,
                    self.ConsumptionCost,
                    inventorycoststochasticperiod,
                    setupcoststochasticperiod,
                    backordercoststochasticperiod,
                    nrsetups,
#                    averagecoverage,
                    evaluationduration
                    ] \
                  + AverageStockAtLevel + [0]*(5-self.Instance.NrLevel) + nrbackorerxperiod + [0]*(49-self.Instance.NrTimeBucket)+[nrlostsale]

        data = testidentifier.GetAsStringList() + [filepostscript, len(self.Scenarioset)] + kpistat
        if Constants.PrintDetailsExcelFiles:
            d = datetime.now()
            date = d.strftime('%m_%d_%Y_%H_%M_%S')
            myfile = open(r'./Test/Statistic/TestResult_%s_%r_%s.csv' % (self.Instance.InstanceName, filepostscript, date), 'w')
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(data)
            myfile.close()
        return kpistat

    #Return the number of setups in the solution
    def GetNrSetup(self):
        result = sum(self.Production[w][t][p] for p in self.Instance.ProductSet for t in self.Instance.TimeBucketSet for w in range( len(self.Scenarioset) ))
        result = result / len(self.Scenarioset)
        return result

    #return the average coverage per oder
    def GetAverageCoverage(self):

        avgdemand = self.Instance.ComputeAverageDemand()

        result = sum(sum(self.ProductionQuantity[w][t][p] / avgdemand[p]
                             for w in range(len(self.Scenarioset))
                          for t in self.Instance.TimeBucketSet if self.ProductionQuantity[w][t][p] )
                      for p in self.Instance.ProductSet)
        nrsetup = self.GetNrSetup() * len(self.Scenarioset)
        result= result / nrsetup

        print("Not defined yet")
        return result

    # This function return the current level of stock and back order based on the quantoty ordered and demands of previous perriod
    def GetCurrentStatus(self, prevdemand, prevquanity, time, projinventorymusbepositive = True):
        projectedinventory = [0 for p in self.Instance.ProductSet]
        projectedbackorder = [0 for p in self.Instance.ProductWithExternalDemand]

        # sum of quantity and initial inventory minus demands
        projinventory = [(self.Instance.StartingInventories[p]
                                  + sum(prevquanity[t][p] for t in range(max(time - self.Instance.LeadTimes[p] + 1 , 0)))
                                  - sum(prevquanity[t][q] * self.Instance.Requirements[q][p] for t in range(time +1) for q in self.Instance.ProductSet)
                                  - sum(prevdemand[t][p] for t in range(time +1)))
                                    for p in self.Instance.ProductSet]

        currentechlonnventory = [(self.Instance.StartingInventories[p]
                                  + sum(prevquanity[t][p] for t in range(time))
                                  - sum(prevquanity[t][q] * self.Instance.Requirements[q][p] for t in range(time) for q in self.Instance.ProductSet)
                                  - sum(prevdemand[t][p] for t in range(time)))
                                    for p in self.Instance.ProductSet]

        currentinventory = [(self.Instance.StartingInventories[p]
                             + sum(prevquanity[t][p] for t in range(max(time - self.Instance.LeadTimes[p] + 1, 0)))
                             - sum(prevquanity[t][q] * self.Instance.Requirements[q][p] for t in range(time) for q in self.Instance.ProductSet)
                             - sum(prevdemand[t][p] for t in range(time)))
                            for p in self.Instance.ProductSet]
        for p in self.Instance.ProductSet:
             if projinventory[p] > - 0.0001: projectedinventory[p] = projinventory[p]
             else:
                 if not self.Instance.HasExternalDemand[p] and not self.NotCompleteSolution and projinventorymusbepositive:
                     print("time: %r " % (time))
                     print("prevquanity: %r " % (prevquanity))
                     print("inventory: %r " % (projinventory))
                     raise NameError(" A product without external demand cannot have backorder")
                     projectedbackorder[ self.Instance.ProductWithExternalDemandIndex[p] ] = -projinventory[p]

        return projectedbackorder, projinventory, currentechlonnventory, currentinventory

    #Return the violation of flow constraint of the suggested quantity
    def ComputeProductViolation( self, suggestedquantities, previousstocklevel ):
        result = [max(sum(self.Instance.Requirements[q][p] * suggestedquantities[q] for q in self.Instance.ProductSet)
                       - previousstocklevel[p], 0.0)
                  if not self.Instance.HasExternalDemand[p]
                  else 0.0
                  for p in self.Instance.ProductSet]
        return result

    # Return the capacity violation of the suggested quantity
    def ComputeResourceViolation(self, suggestedquantities, previousstocklevel):
        result = [max(sum(self.Instance.ProcessingTime[q][k]*suggestedquantities[q] for q in self.Instance.ProductSet)
                       - self.Instance.Capacity[k], 0.0)
                  for k in self.Instance.ResourceSet]
        return result

    #return the slack in the flow conservation constraint associated with the suggested quantities
    def getProductionCostraintSlack(self,  suggestedquantities, previousstocklevel ):
        result = [max(previousstocklevel[p] - sum(self.Instance.Requirements[q][p] * suggestedquantities[q]
                                                   for q in self.Instance.ProductSet)
                      ,0.0)
                                                    for p in self.Instance.ProductSet]
        return result

    # return the slack in the capacity constraint associated with the suggested quantities
    def getCapacityCostraintSlack(self, suggestedquantities ):
        result = [max(self.Instance.Capacity[k] - sum(self.Instance.ProcessingTime[q][k] * suggestedquantities[q]
                                                        for q in self.Instance.ProductSet),
                       0.0) for k in self.Instance.ResourceSet]
        return result

    #reutn the possible increase in production quantity that respect the flow conservation and capacity constraints
    def ComputeAvailableFulliment(self, product,  productionslack, capacityslack ):
        maxcomponent = min(productionslack[p]/ self.Instance.Requirements[product][p]
                            if self.Instance.Requirements[product][p] > 0
                            else Constants.Infinity
                         for p in self.Instance.ProductSet)

        maxresource = min(capacityslack[k]/self.Instance.ProcessingTime[product][k]
                            if self.Instance.ProcessingTime[product][k] > 0 else Constants.Infinity
                          for k in self.Instance.ResourceSet)
        result = min(maxcomponent, maxresource)
        return result

    #This function adjust the quantities, to respect the flow constraint
    def RepairQuantityToOrder(self, suggestedquantities, previousstocklevel):

        idealquuantities = [suggestedquantities[p] for p in self.Instance.ProductSet]
        #Compute the viiolation of the flow constraint for each component
        productviolations = self.ComputeProductViolation(suggestedquantities, previousstocklevel)
        productmaxvioalation = np.argmax(productviolations)
        maxproductviolation = productviolations[productmaxvioalation]

        resourceviolations = self.ComputeResourceViolation(suggestedquantities, previousstocklevel)
        resourcemaxvioalation = np.argmax(resourceviolations)
        maxresourceviolation = resourceviolations[resourcemaxvioalation]
        maxviolation = max(maxresourceviolation, maxproductviolation)
        isproductviolation = maxviolation == maxproductviolation
        #While some flow constraints are violated, adjust the quantity to repect the most violated constraint
        while( maxviolation > 0.000001 ) :
            if Constants.Debug:
                print(" the max violation %r is from %r " %(maxviolation, productmaxvioalation ))
                print(" quantities: %r " % (suggestedquantities))

            if isproductviolation:
                producyqithrequirement = [p for p in self.Instance.ProductSet
                                           if self.Instance.Requirements[p][productmaxvioalation] > 0]
                totaldemand = sum(self.Instance.Requirements[q][productmaxvioalation]*suggestedquantities[q]
                                   for q in self.Instance.ProductSet)
                ratiodemande = [self.Instance.Requirements[q][productmaxvioalation]*suggestedquantities[q]/totaldemand
                                 for q in self.Instance.ProductSet]
            else:
                producyqithrequirement = [p for p in self.Instance.ProductSet if
                                          self.Instance.ProcessingTime[p][resourcemaxvioalation] > 0]
                totaldemand = sum(self.Instance.ProcessingTime[q][resourcemaxvioalation] * suggestedquantities[q]
                                   for q in self.Instance.ProductSet)
                ratiodemande = [self.Instance.ProcessingTime[q][resourcemaxvioalation] * suggestedquantities[q] / totaldemand
                                 for q in self.Instance.ProductSet]

            for p in producyqithrequirement:
                quantitytoremove = (1.0*maxviolation) * ratiodemande[p]
                suggestedquantities[p] = max(suggestedquantities[p] - quantitytoremove, 0)

            if Constants.Debug:
                print(" new quantities: %r " %(suggestedquantities))

            productviolations = self.ComputeProductViolation(suggestedquantities, previousstocklevel)
            productmaxvioalation = np.argmax(productviolations)
            maxproductviolation = productviolations[productmaxvioalation]

            resourceviolations = self.ComputeResourceViolation(suggestedquantities, previousstocklevel)
            resourcemaxvioalation = np.argmax(resourceviolations)
            maxresourceviolation = resourceviolations[resourcemaxvioalation]
            maxviolation = max(maxresourceviolation, maxproductviolation)

            isproductviolation = maxviolation == maxproductviolation

        for p in self.Instance.ProductSet:
            productionslack = self.getProductionCostraintSlack(suggestedquantities, previousstocklevel)
            capacityslack = self.getCapacityCostraintSlack(suggestedquantities)
            suggestedquantities[p] = suggestedquantities[p] + min(idealquuantities[p] - suggestedquantities[p],
                                                                       self.ComputeAvailableFulliment(p, productionslack, capacityslack) )


    # This function return the quantity to order a time t, given the first t-1 demands with the S policy
    def GetQuantityToOrderS(self, time, previousdemands, previousquantity=[]):

        previousdemands2 = previousdemands+[[0.0 for p in  self.Instance.ProductSet]]+[[0 for p in  self.Instance.ProductSet]]
        projectedbackorder, projectedstocklevelatstart, currrentechelonstocklevel, currentinventory = self.GetCurrentStatus(previousdemands2, previousquantity, time)

        previousquantity2 =  [[ previousquantity[t][p] for p in self.Instance.ProductSet] for t in self.Instance.TimeBucketSet]

        quantity = [ 0  for p in self.Instance.ProductSet]

        level = [ self.Instance.Level[p] for p in self.Instance.ProductSet]
        levelset = sorted(set(level), reverse=False)
        for l in levelset:
            prodinlevel = [p for p in self.Instance.ProductSet if self.Instance.Level[p]== l]
            for p in prodinlevel:
                if self.Production[0][time][p] >= 0.9:

                          projectedbackorder, projectedstocklevel, currrentechelonstocklevel, currrentstocklevel2 \
                              = self.GetCurrentStatus( previousdemands2, previousquantity2, time , projinventorymusbepositive= False)
                          echelonstock = Tool.ComputeInventoryEchelon(self.Instance, p, currrentechelonstocklevel)

                          quantity[p] = max(self.SValue[time ][p] - echelonstock, 0)

                          self.RepairQuantityToOrder(quantity, currentinventory)
                          previousquantity2[time][p] = quantity[p]

        error = 0
        return quantity, error

    #This function merge solution2 into self. Assume that solution2 has a single scenario
    def Merge( self, solution2 ):
        self.Scenarioset.append( solution2.Scenarioset[0] )
        self.SenarioNrset = range(len(self.Scenarioset))
        self.ProductionQuantity = self.ProductionQuantity + solution2.ProductionQuantity
        self.InventoryLevel = self.InventoryLevel + solution2.InventoryLevel
        self.Production = self.Production + solution2.Production
        self.BackOrder = self.BackOrder + solution2.BackOrder
        self.Consumption = self.Consumption + solution2.Consumption

    #Compute the average value of S (see paper)
    def ComputeS( self ):
        S = [ [0 for p in self.Instance.ProductSet ] for t in self.Instance.TimeBucketSet ]
        SWithLeftover = [ [0 for p in self.Instance.ProductSet ] for t in self.Instance.TimeBucketSet ]
        probatime = [ [0 for p in self.Instance.ProductSet ] for t in self.Instance.TimeBucketSet ]

        nrconsideredsol = [ [0 for p in self.Instance.ProductSet ] for t in self.Instance.TimeBucketSet ]
        for w in range( len(self.Scenarioset) ):
            s =self.Scenarioset[w]
            for n in s.Nodes:
                    t= n.Time
                    for p in self.Instance.ProductSet:
                        if t < self.Instance.NrTimeBucket and (self.Production[w][t][p] >=0.9):
                            if n.HasLeftOverComponent(p)and n.HasSpareCapacity(p):
                                nrconsideredsol[t][p] += s.Probability
                                SWithLeftover[t][p] += n.GetS(p) * s.Probability

                            if n.GetS(p) > S[t][p]:
                                S[t][p] = n.GetS(p)

                            probatime[t][p] = probatime[t][p] + s.Probability

        SValueBasedOnMAx = [[S[t][p] if probatime[t][p] > 0 else 0.0
                             for p in self.Instance.ProductSet]
                            for t in self.Instance.TimeBucketSet ]

        self.SValue = [[SWithLeftover[t][p]/nrconsideredsol[t][p] if nrconsideredsol[t][p] > 0 else SValueBasedOnMAx[t][p]
                        for p in self.Instance.ProductSet]
                       for t in self.Instance.TimeBucketSet]

        if Constants.Debug:
            print("The value of S is: %r" % (self.SValue))

    #return the scenario tree of the average demand
    @staticmethod
    def GetAverageDemandScenarioTree(instance):
        scenariotree = ScenarioTree( instance,
                                     [1]*(instance.NrTimeBucket+1) + [0],
                                     0,
                                     averagescenariotree=True,
                                     scenariogenerationmethod=Constants.MonteCarlo,
                                     model = "YQFix" )
        return scenariotree

    #Create an empty solution (all decisions = 0) for the problem
    @staticmethod
    def GetEmptySolution( instance ):
        scenariotree = Solution.GetAverageDemandScenarioTree( instance )
        scenarioset = scenariotree.GetAllScenarios(False)
        production = [[[0 for p in instance.ProductSet] for t in instance.TimeBucketSet] for w in scenarioset]
        quanitity = [[[0 for p in instance.ProductSet] for t in instance.TimeBucketSet] for w in scenarioset]
        stock = [[[0 for p in instance.ProductSet] for t in instance.TimeBucketSet] for w in scenarioset]
        backorder = [[[0 for p in instance.ProductWithExternalDemand] for t in instance.TimeBucketSet] for w in scenarioset]
        consumption = [[[[0 for p in instance.ProductSet] for q in instance.ProductSet] for t in instance.TimeBucketSet] for w in scenarioset]
        result = Solution(instance=instance,
                          scenriotree=scenariotree,
                          scenarioset=scenarioset,
                          solquantity=quanitity,
                          solproduction=production,
                          solbackorder=backorder,
                          solconsumption=consumption,
                          solinventory=stock)

        result.NotCompleteSolution = True
        result.SValue =[[0 for p in instance.ProductSet] for t in instance.TimeBucketSet]

        result.FixedQuantity= [[0 for p in instance.ProductSet] for t in instance.TimeBucketSet]
        return result

    def ComputeInventory(self):
        for w in self.SenarioNrset:
            for t in self.Instance.TimeBucketSet:
                prevdemand = [[self.Scenarioset[w].Demands[tau][q] for q in self.Instance.ProductSet] for tau in range(t+1)]
                prevquanty = [[self.ProductionQuantity[w][tau][q] for q in self.Instance.ProductSet] for tau in range(t+1)]
                currentinventory = [(self.Instance.StartingInventories[p]
                                         + sum(prevquanty[t][p] for t in range(max(t - self.Instance.LeadTimes[p] + 1, 0)))
                                         - sum(self.Consumption[w][t][c[0]][c[1]] for t in range(t+1) for c in self.Instance.ConsumptionSet if c[0]==p)
                                         - sum(prevdemand[t][p] for t in range(t+1)))
                                        for p in self.Instance.ProductSet]

                for p in self.Instance.ProductSet:

                    if currentinventory[p] >= -0.0001:
                        self.InventoryLevel[w][t][p] = currentinventory[p]
                        if self.Instance.HasExternalDemand[p]:
                            indexp = self.Instance.ProductWithExternalDemandIndex[p]
                            self.BackOrder[w][t][indexp] = 0.0
                    else:
                        if self.Instance.HasExternalDemand[p]:
                            indexp = self.Instance.ProductWithExternalDemandIndex[p]
                            self.BackOrder[w][t][indexp] = -currentinventory[p]
                            self.InventoryLevel[w][t][p] = 0.0
                        else:
                            self.InventoryLevel[w][t][p] = currentinventory[p]
                            if Constants.Debug:
                                print("THE SOlution is not feasible!!!!!!")