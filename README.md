This git repository provides the code associated with the paper entitled
<strong>"Stochastic Dual Dynamic Programming for Multi-Echelon Lot-sizing with Component Substitution."</strong> 
by <em>Simon Thevenin, Yossiri Adulyasak, and Jean-François Cordeau</em>. 

<br/>
This code is provided to facilitate the reproducibility of the results and to share the datasets. 

<h1> Prerequisits </h1>

The code must be run with Python 2, with all the required libraries (see the import section at the beginning of each python file) installed. The softwares <em>Cplex</em> and <em>LatexBuilder</em> must be installed.



<h1> To run the software </h1>

The file scm.py is the main entry point to the program.  The program can be run with the following command line:
<br/>
<pre><code> Solve instance_name model nr_scenario scenario_sampling_method </code></pre>
 <br/>

The parameter <strong>instance_name</strong> refers to the name of the instance to solve. The instance must be provided as a file in the correct format (see the examples) in the folder Instance.  

The parameter <strong>model</strong> provides the type of stochastic model to consider for the instance. 
-	YFix refers to the static-dynamic model where the setups are decided in period 0 and fixed for the entire horizon, whereas the production quantities are decided dynamically when new information on the demand is available.
-	YQFix refers to the static model where the setups and the production quantities are decided in period 0 and frozen for the entire horizon.
-	Average refers to the deterministic model where the demand takes its expected value. 
-	HeuristicYFix refers to a solution strategy to solve the YFix model, where we first solve the 2-stage problem to fix the setup decisons, before to solve the multi-stage problem on a large scenario tree.

<br/>
The parameter <strong>nr_scenario</strong>  gives the number of scenarios used to optimize the problem. When solving YFix or HeuristicYFix, the value must be chosen within predefined tree shapes such as "all5" for a tree with 5 scenarios per branch. For the complete list of predefined tree structures (or for defining additional tree structures), see the function <em>GetTreeStructure()</em> in the file <em>Solver.py</em>.

The parameter <strong>scenario_sampling_method</strong> refers to the scenario sampling approach, and its value may be: 
-	MC for crude Monte Carlo.
-	RQMC for randomized Quasi Monte Carlo.
<br/>

For example, the following command solves the instance SuperSmallIntance.xlsx for a multi-stage program, where the scenario tree has 2 branches per node, and the demand values are sampled with RQMC. 
<pre><code>python scm.py Solve SuperSmallIntance YFix all2 RQMC </code></pre>
Once the instance is solved, a summary of the results is available in the folder Test. To have a detailed description of the solution in the folder <em>Solution.py</em>, turn the parameter <em>PrintSolutionFileToExcel</em> in the file  <em>Constant.py</em> to True.


The option “-m” allows selecting the <strong>optimization approach</strong> , and this parameter may take the values: 
-	SDDP for the Stochastic Dual Dynamic Programming approach
-	PH for the Progressive Hedging Method
-	Hybrid For the Hybrid of PH and SDDP
-	MLLocalSearch for the heuristic version of SDDP

The paper provides a description of each these methods. 

After solving the instance, the program performs a <strong>simulation </strong>to evaluate the performance of the approach in a stochastic environment. The number of scenarios used in the evaluation can be tunned with the parameter “-n” (e.g., “-n 5000”).

Other options are available, and they are described in the function <em>parseArguments</em> of the file <em>scm.py</em>

<h1> Short description of the class structure </h1>

This section provide a brief overview of the class structur that model each optimization method provided in the paper.

<h2> Large MILP based on a scenario tree  </h2>

<em>MIPSolver.py</em>  contains the code that creates an object MIPSolver with the attributes and methods to create a big MILP based on a scenario tree to solve the Multi echelon Lot Sizing problem under Demand uncertainty. Depending on the shape of the scenario tree, <em>MIPSolver.py</em>  can also solve a deterministic or a two-stage model. The class ScenarioTree, ScenarioTreeNode, and Scenario define the structure of the scenario tree required to build a multi-stage stochastic program.

<h2> Stochastic dual dynamic programming  </h2>
<em>SDDP.py</em> contains the code to create an object SDDP.  The object SDDP contains methods to run the algorithm, such as methods for the Forward and backward pass. The object SDDP has a link toward several object SDDPStage that contains the CPLEX model of each decision stage.

<em>SDDPStage.py</em> is an object that contains the CPLEX model for a specific decision stage in SDDP. The cuts that build the approximation of the cost-to-go are stored in the object <em>SDDPCut.py</em>. As the last stage cut requires a special handling, it is stored in an object  <em>SDDPLastStageCut.py</em>. 

The class <em>MLLocalSearch.py</em> contains the attributes and methods allowing to define an heuristic SDDP approach. In each iteration, the stage 2 to T are solved to optimality, and the first stage is solve to improve the setup decisions.

The git repository also includes the code of some negative results that are not included in the paper, such as <em>SDDPCallBack.py</em>  and <em>SDDPUserCutCallBack.py</em>  to add the cuts “on the fly” in the branching tree of CPLEX, and <em>SDDPML.py</em>  to builds cuts by approximating the cost to go with machine learning tools.

<h2> Progressive hedging  </h2>
<em>ProgressiveHedging.py</em>  describes an object that runs the PH method, and <em>Hybrid_PH_SDDP.py</em> describes an object that runs the hybrid PH and SDDP method (see the paper).

<h2> Simulation </h2>
The git repository also includes the code to evaluate the approach with a simulation. The class <em>EvaluationSimulator.py</em> provides a framework to evaluate the performance of the methods through a simulation over a large number of scenarios. The class <em>Evaluator.py</em> contains the method to call the simulator and run the evaluation. The object <em>EvaluatorIdentificator.py</em> contains all the information to identify the evaluator, and this information is recorded in the results. Such objects are used to record the specific experimental design used during a test (the number of simulation scenarios, the time horizon, and the policy used to update decisions when new information is available.).

<h2> Instances </h2>
<em>Instance.py</em> is an object that describes a specific instance of the capacitated multi-echelon lot sizing problem with component substitution under demand uncertainty. The object <em>Solution.py</em>  describes a solution (a production plan), and it provides methods to save and read solutions from files.



<em>InstanceReader.py</em>  defines the methods to read an instance from a file. The goal is to generate instances based on files existing from the literature. As several input formats exist, the class can be specialized in several children classes (one for each type of input file): <em>InstanceReaderGrave.py</em> and <em>InstanceReaderTemplemeyer.py</em>.


<h2> Other files </h2>
<em>Constant.py</em> list some parameters of the methods (the time limit of the algorithm, tolerance, ...), as well as various constants used througout the code (e.g., the name of the methods).

Note that the code was built upon the code associated with the paper from Thevenin et al (2021) . 
As a results some files (such as DecentralizedMRP.Py or RollingHorizonSolver.py) are not used in the results presented in the paper.
<br/>
<br/>

<em>Thevenin et al (2021): Thevenin, S., Adulyasak, Y., & Cordeau, J. F. (2021). <a href="[http://www.google.com](https://onlinelibrary.wiley.com/doi/10.1111/poms.13277)" title="Material requirements planning under demand uncertainty using stochastic optimization. Production and Operations Management">Material requirements planning under demand uncertainty using stochastic optimization. Production and Operations Management</a>. Production and Operations Management, 30(2), 475-493.</em>


