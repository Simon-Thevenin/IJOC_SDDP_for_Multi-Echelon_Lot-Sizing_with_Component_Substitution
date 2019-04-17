from __future__ import absolute_import, division, print_function

import pandas as pd

class Scenario(object):

    NrScenario = 0
    #Constructor
    def __init__( self, owner = None, demand = None, proabability = -1, quantityvariable = None,  productionvariable = None,  inventoryvariable = None,  backordervariable = None, consumptionvariable = None, nodesofscenario = None ):
        self.Owner = owner
        # The  demand in the scenario for each time period
        self.Demands = demand
        # The probability of the partial scenario
        self.Probability = proabability
        # The attribute below contains the index of the CPLEX variables (quanity, production, invenotry) for each product and time.
        self.QuanitityVariable = quantityvariable
        self.ProductionVariable = productionvariable
        self.InventoryVariable = inventoryvariable
        self.BackOrderVariable = backordervariable
        self.ConsumptionVariable = consumptionvariable
        Scenario.NrScenario = Scenario.NrScenario +1
        self.Nodes = nodesofscenario
        if not self.Nodes is None:
            for n in self.Nodes:
                n.OneOfScenario = self
                n.Scenarios.append(self)
        self.ScenarioId = Scenario.NrScenario

    def DisplayScenario(self):
        print("Scenario %d" %self.ScenarioId)
        print("Demand of scenario( %d ): %r" %(self.ScenarioId, self.Demands))
        print("Probability of scenario( %d ): %r" %(self.ScenarioId, self.Probability))
        print("Quantity variable of scenario( %d ): %r" % (self.ScenarioId, self.QuanitityVariable))
        print("Production variable of scenario( %d ): %r" % (self.ScenarioId, self.ProductionVariable))
        print("Consumption variable of scenario( %d ): %r" % (self.ScenarioId, self.ConsumptionVariable))

        print("Inventory variable of scenario( %d ): %r" % (self.ScenarioId, self.InventoryVariable))
        print("BackOrder variable of scenario( %d ): %r" % (self.ScenarioId, self.BackOrderVariable))

    # This function print the scenario in an Excel file in the folde "Solutions"
    def PrintScenarioToExcel(self, writer):
        demanddf = pd.DataFrame(self.Demands, columns = self.Owner.Instance.ProductName, index=self.Owner.Instance.TimeBucketSet)
        demanddf.to_excel(writer, "DemandScenario %d" %self.ScenarioId)
        quantitydf = pd.DataFrame(self.QuanitityVariable, columns=self.Owner.Instance.ProductName, index=self.Owner.Instance.TimeBucketSet)
        quantitydf.to_excel(writer, "QuanitityVariable %d" %self.ScenarioId)
        productiondf = pd.DataFrame(self.ProductionVariable, columns=self.Owner.Instance.ProductName, index=self.Owner.Instance.TimeBucketSet)
        productiondf.to_excel(writer, "ProductionVariable %d" % self.ScenarioId)
        inventorydf = pd.DataFrame(self.InventoryVariable, columns=self.Owner.Instance.ProductName, index=self.Owner.Instance.TimeBucketSet)
        inventorydf.to_excel(writer, "InventoryVariable %d" % self.ScenarioId)
        bbackorderydf = pd.DataFrame(self.BackOrderVariable, columns=self.Owner.Instance.ProductName, index=self.Owner.Instance.TimeBucketSet)
        bbackorderydf.to_excel(writer, "BackOrderVariable %d" % self.ScenarioId)

        indexprodprod = pd.MultiIndex.from_tuples([self.Owner.Instance.ProductName, self.Owner.Instance.ProductName], names=['prodfrom', 'prodto'])
        consumptionvardf = pd.DataFrame(self.ConsuptionVariable, columns=indexprodprod, index=self.Owner.Instance.TimeBucketSet)
        consumptionvardf.to_excel(writer, "ConsuptionVariable %d" % self.ScenarioId)