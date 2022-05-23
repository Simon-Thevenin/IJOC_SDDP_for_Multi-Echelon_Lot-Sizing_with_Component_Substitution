This git repository provide the code associated with the paper entitled
<strong>"Stochastic Dual Dynamic Programming for Multi-Echelon Lot-sizing with Component Substitution."</strong> 
From  <em>Simon Thevenin, Yossiri Adulyasak, and Jean-François Cordeau</em>. 

<br/>
This code is provided to facilitate the reproducibility of the results and to share the datasets. 

<h1> Prerequisits </h1>

The code must be run with Python 2, and the softwares <em>Cplex</em> and <em>LatexBuilder</em> must be installed.
Note that the code was built upon the code associated with the paper from Thevenin et al (2021) . 
As a results some files (such as DecentralizedMRP.Py or RollingHorizonSolver.py are not used in the results of the paper)
<br/>
<br/>

<em>Thevenin et al (2021): Thevenin, S., Adulyasak, Y., & Cordeau, J. F. (2021). Material requirements planning under demand uncertainty using stochastic optimization. Production and Operations Management, 30(2), 475-493.</em>


<h1> To run the software </h1>

The file scm.py is the main entrance point to the program. 


Python 
The file scm.py is the main entrance point to the program. 

Solve instance_name model nrscenario scenariosampling -n 5000 -p Re-solve -m Hybrid --mipsetting Default  --hybridphsetting Multiplier1
Instance_name refers to the name of the instance to solve. The instance must be provided as a file in the right format (see the examples) in the folder Instance.  
Model provide the type of stochastic model to consider for the instance. 
-	YFix refers to the static-dynamic model where the setup are decided in period 0 and fixed for the entire horizon, wwheras the production quantity are decided dynamically when new information on the demand is available.
-	YQFix refers to the static model where the setup and the production quantity are decided in period 0 and frozen for the entire horizon.
-	Average refers to the deterministic model where the demand takes its expected value. 
-	HeuristicYFix refers to a solution strategy to solve the YFix model, where we first solve the 2-stage problem to fix the Y variables, then solve the multi-stages problem on large scenario tree.
Nrscenario give the number of scenario used to optimize the problem. When solving YFix or HeuristicYFix, the value must be chosen within predefined tree shapes such as "all5" for a tree with 5 scenarios per branch. For the complete list of predefined tree structure (or for defining additional tree structur). See the function GetTreeStructure() in the file Solver.py.

Scenariosampling refers to the scenario sampling approach, and the value may be: 
-	MC for crude Monte Carlo
-	RQMC for randomized Quasi Monte Carlo
For instance, the command 
“python scm.py Solve SuperSmallIntance YFix all5 RQMC” 
Solve the instance SuperSmallIntance.xlsx for a multi-stage program where the scenario tree has 5 branches per nodes, and the demand values are sampled with RQPC. Once the instance is solved, a summary of the results is available in the folder Test. To have a detailed description of the solution in the folder “Solution”, turn the parameter “PrintSolutionFileToExcel” in Constant.py to True.

The option “-m” allow selecting the optimization approach, and this parameter may take value: 
-	SDDP for the Stochastic Dual Dynamic Programming approach
-	PH for the Progressive Hedging Method
-	Hybrid For the Hybrid of PH and SDDP
-	MLLocalSearch for the heuristic version of SDDP
A description of these methods is provided in the paper. 


After solving the instance, the program performs a simulation to evaluate the performance of the approach in a stochastic environment. The number of scenarios used in the evaluation can be tunned with the parameter “-n” (e.g., “-n 5000”), and the 

	

<h1> Short description of the class structure </h1>

MIPSolver.py contains the code that create an object MIPSolver with the attributes and methods to create a big MILP based on a scenario tree to solve the Multi echelon Lot Sizing problem under Demand uncertainty. Depending on the shape of the scenario tree, MIPSolver.py can also solve a deterministic or a two-stage model.The class scenarioTree, ScenarioTreeNode, and Scenario define the structure of the scenario tree required to build a multi-stage stochastic program.
SDDP.py contains the code to create an object SDDP. The object SDDP refers to several object SDDPStage that contains the CPLEX model of the corresponding decision stage. The object SDDP contain methods to run the algorithm, such as methods for the Forward and backward pass. SDDPStage.py is an object that contains the CPLEX model for a specific decision stage in SDDP.  The cuts that build the approximation of the cost-to-go are stored in the object SDDPCut. As the last stage cut requires a special handling, it is stored in an object SDDPLastStageCut. The github repository also include the code of some negative results that are not included in the paper, such as SDDPCallBack and SDDPUserCutCallBack to add the cuts “on the fly” in the branching tree of CPLEX, SDDPML to builds cuts by approximating the cost to go with machine learning tools.
The class MLLocalSearch contains the attributes and methods allowing to define an iterative SDDP approach. In each iteration, the stage 2 to T are solved to optimality, and the first stage is solve to improve the setup decisions.

ProgressiveHedging.py describes an object that runs the PH method. Hybrid_PH_SDDP.py describes an object that run the hybrid PH and SDDP method (see the paper).

The git repository also include the code to evaluate the approach with a simulation. The class EvaluationSimulator.py provides a framework to evaluate the performance of the method through a simulation over a large number of scenarios. The class Evaluator.py contains the method to call the simulator and run the evauationThe object EvaluatorIdentificator contains all the information which allow to identify the evaluator. Such objects are used to record the specific experimental design used during a test (the number of simulation scenario, the time horizon, the policy used to update decision when new information is available.).

Instance.py is an object that describes a specific instance of the capacitated multi-echelon lot sizing problem with component substitution under demand uncertainty. The object Solution.py describes a solution (a production plan)

InstanceReader.py define the methods to read an instance from a file. The goal is to generate instances based  on files existing from the literature. As several input format exists, the class can be specialized in several children class (on for each type of input file): InstanceReaderGrave.py, InstanceReaderTemplemeyer.py, and InstanceReaderJDA.py
	
Constant.py contains many constants variable used in the code.

TO BE DELETED: 
-	MIPSolverMultiStage
-	MIPSolverTwoStage

