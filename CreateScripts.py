#This file generates the scripts to run all the instances on server grids.
#The script are specific for each computer grid.
#The file InstancesToSolve.csv must be filled with the instance on which the code will be run.
# The file takes as input a parameter to describe the type of test to run. Use:
#   - H for a comparison of the different heuritic to solve the multi stage problem
#   -SDDP to test the performance of SDDP when some improvement are removed.
#   -SDDPScenario to test SDDP with various number of scenario
#   - TestMultiplierPH to tune the value of the lagrangian multiplier in PH
#   - N to compare the performance of a large MIP based on a scenario tree and SDDP

import sys
import csv
from Constants import Constants

NrScenarioEvaluation = "5000"
ForCIRRELT = False

def CreatHeader(file):
       CreatHeaderNantes(file)


def CreatHeaderCirrelt(file):
    file.write("""#!/bin/bash -l
#
#$ -cwd
#$ -q idra
#$ -j y
mkdir -p /tmp/thesim
mkdir -p /tmp/thesim/Evaluations
mkdir -p /tmp/thesim/Solutions
mkdir -p /tmp/thesim/CPLEXLog
""")

def CreatHeaderQuebec(file):
    file.write("""#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --job-name=test
#SBATCH --output=./Temp/%x-%j.out
#SBATCH --mem=10G
mkdir /tmp/thesim

module load python/2.7
module load scipy-stack
module load python/2.7
module load cplex/2.9


ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
virtualenv --no-download -p /home/LS2N/thevenin-s/python/bin/python2.7 $ENVDIR

source $ENVDIR/bin/activate
pip install --upgrade pip
cd /home/thesim/installcplex/CPLEX_Studio129/cplex/python/2.7/x86-64_linux/
python setup.py install
cd /home/thesim/
pip install openpyxl --upgrade --pre
pip install networkx
pip install sklearn
module load scipy-stack


pip freeze > requirements.txt


cd /home/thesim/ProjetJFY/scm


mkdir -p /tmp/thesim
mkdir -p /tmp/thesim/Evaluations
mkdir -p /tmp/thesim/Solutions
mkdir -p /tmp/thesim/CPLEXLog

""")


def CreatHeaderNantes(file):
    file.write("""#!/bin/bash
# Nom du job
#SBATCH -J MON_JOB_MPI
#
# Partition visee
#SBATCH --partition=SMP-medium
#
# Nombre de noeuds
#SBATCH --nodes=1
# Nombre de processus MPI par noeud
#SBATCH --ntasks-per-node=1
#SBATCH --mem 40000
#
# Temps de presence du job
#SBATCH --time=40:00:00
#
# Adresse mel de l'utilisateur
#
# Envoi des mails
#SBATCH --mail-type=abort,end
#SBATCH --mail-user=simon.thevenin@imt-atlantique.fr
#
#SBATCH -o /home/LS2N/thevenin-s/log/job_mpi-%j.out
 
module purge
module load intel/2016.3.210
module load intel/mkl/64/2016.3.210
module load intel/mpi/2016.3.210
module load python/2.7.12
module load intel/mkl/64/2017.4.196
module load compilateurs_interpreteurs/gcc/7.3.0

export LD_PRELOAD=/lib64/psm2-compat/libpsm_infinipath.so.1



#
# Faire le lien entre SLURM et Intel MPI
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

""")

#SBATCH --mail-user=simon.thevenin@imt-atlantique.fr

def CreateSDDPJob(instance, nrback, nrforward, setting, model = "YFix", nrtest="5000"):
    qsub_filename = "./Jobs/job_sddp_%s_%s_%s_%s_%s" % (instance, nrback, nrforward, setting, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    scengen = "RQMC"
    if setting == "MC" or setting == "NoEnhancement":
        scengen = "MC"
    qsub_file.write("""
srun python scm.py  Solve %s %s %s %s -n %s -p Re-solve -m SDDP --sddpsetting %s --nrforward %s  > /home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt 
""" % (instance, model, nrback, scengen, nrtest, setting, nrforward))
    return qsub_filename

def CreateMLLocalSearchJob(instance, nrback, nrforward, setting, model = "YFix", mlsetting = "Defuault", sddpsetting = "Default"):
    qsub_filename = "./Jobs/job_mllocalsearch_%s_%s_%s_%s_%s_%s" % (instance, nrback, nrforward, setting, model, mlsetting)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
srun python scm.py  Solve %s %s %s RQMC -n 5000 -p Re-solve -m MLLocalSearch --mipsetting %s --mllocalsearchsetting %s --nrforward %s --sddpsetting %s>/home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt
""" % (instance, model, nrback, setting, mlsetting, nrforward, sddpsetting))
    return qsub_filename

def CreateHybridSearchJob(instance, nrback, nrforward, setting, model = "YFix", phsetting = "Default"):
    qsub_filename = "./Jobs/job_hybrid_%s_%s_%s_%s_%s_%s" % (instance, nrback, nrforward, setting, model, phsetting)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
srun python scm.py  Solve %s %s %s RQMC -n 5000 -p Re-solve -m Hybrid --mipsetting %s  --hybridphsetting %s --nrforward %s   >/home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt
""" % (instance, model, nrback, setting, phsetting, nrforward))
    return qsub_filename

def CreateMIPJob(instance, scenariotree, model = "YFix", mipsetting = "Default", evaluationpolicy = "Re-solve" ):
    qsub_filename = "./Jobs/job_mip_%s_%s_%s_%s_%s" % (instance, scenariotree, model, mipsetting, evaluationpolicy)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file)
    qsub_file.write("""
srun python scm.py  Solve %s %s %s RQMC -n 5000 -p %s -m MIP --mipsetting %s  >/home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt 
""" % (instance, model, scenariotree, evaluationpolicy, mipsetting))
    return qsub_filename


def CreatePHJob(instance, scenariotree, model = "YFix"):
    qsub_filename = "./Jobs/job_ph_%s_%s_%s" % (instance, scenariotree, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file)
    qsub_file.write("""
srun python scm.py  Solve %s %s %s RQMC -n 5000 -p Re-solve -m PH  >/home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt
""" % (instance, model, scenariotree))
    return qsub_filename

if __name__ == "__main__":
    csvfile = open("./Instances/InstancesToSolve.csv", 'rb')
    data_reader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
    instancenameslist = []
    for row in data_reader:
       instancenameslist.append(row)
    InstanceSet = instancenameslist[0]
    print(InstanceSet)
    instancetosolvename = ""

    scenariotreeset = ["6400b"] #["all2", "all5"]#, "allDIX", "all20"]
    #sddpnrbackset = ["allDIX", "all20", "50-50-10", "all50" ]#,10,20] #[2, 5], 10, 20]
    sddpnrbackset = ["DependOnH"]

    if sys.argv[1] == "N":
        sddpnrbackset = ["all2", "all5", "all10", "all20"]
        filenewname = "runallnewtest.sh"
        filenew = open(filenewname, 'w')
        filenew.write("""
    #!/bin/bash -l
    #
    """)
      #  InstanceSet =["G0041111_Lumpy_b2_fe25_el_rk25_ll0_l20_H04_c2_A0_a0.1","G0041111_Lumpy_b2_fe25_el_rk25_ll0_l20_H02_c2_A2_a0.1","G0041111_Lumpy_b2_fe25_el_rk25_ll0_l20_H06_c2_A2_a1","G0041111_Lumpy_b2_fe25_el_rk25_ll0_l20_H08_c2_A4_a1","G0041111_Lumpy_b2_fe25_el_rk25_ll0_l20_H010_c2_A6_a1"]
        for instance in InstanceSet:
            for nrback in sddpnrbackset:
                for setting in ["SymetricMIP"]:#, "Default"]:
                    nrforward = 1
                    if( not ( "H06" in instance and nrback == "all10"
                              or "H08" in instance and nrback == "all10"
                              or "H010" in instance and nrback == "all10"
                              or "H08" in instance and nrback == "all5"
                              or "H010" in instance and nrback == "all5"
                    ) ) :
                            jobname = CreateMIPJob(instance, nrback, model="YFix", mipsetting=setting)
                            filenew.write("sbatch %s \n" % (jobname))
                            jobname = CreateMLLocalSearchJob(instance, nrback, nrforward, setting, model="YFix", mlsetting="NrIterationBeforeTabu1000")
                            filenew.write("sbatch %s \n" % (jobname))

                    jobname = CreateMLLocalSearchJob(instance, nrback, nrforward, "SymetricMIP", model="YFix",
                                                     mlsetting="NrIterationBeforeTabu1000", sddpsetting = "EvalOutSample")
                    filenew.write("sbatch %s \n" % (jobname))

    if sys.argv[1] == "TestMultiplierPH":
            fileheurname = "multph.sh"
            fileheur = open(fileheurname, 'w')
            fileheur.write("""
#!/bin/bash -l
#
""")

            for instance in InstanceSet:
                for nrback in sddpnrbackset:
                    for setting in ["Default"]:
                        nrforward = 1
                        for phsetting in ["Multiplier00001", "Multiplier0001","Multiplier01", "Multiplier001", "Multiplier1", "Multiplier10"]:
                                jobname = CreateHybridSearchJob(instance, nrback, nrforward, setting, model="YFix", phsetting = phsetting)
                                fileheur.write("sbatch %s \n" % (jobname))



    if sys.argv[1] == "H":
        fileheurname = "runallheur.sh"
        fileheur = open(fileheurname, 'w')
        fileheur.write("""
#!/bin/bash -l
#
""")


        for instance in InstanceSet:
                for nrback in sddpnrbackset:
                    for setting in ["Default"]:
                        nrforward = 1

                        jobname = CreateSDDPJob(instance, nrback, nrforward, setting, model="HeuristicYFix")
                        fileheur.write("sbatch %s \n" % (jobname))
                        jobname = CreateMLLocalSearchJob(instance, nrback, nrforward, setting, model="YFix",mlsetting="NrIterationBeforeTabu1000")
                        fileheur.write("sbatch %s \n" % (jobname))
                        jobname = CreateMIPJob(instance, 100, model="YQFix")
                        fileheur.write("sbatch %s \n" % (jobname))
                        jobname = CreateMIPJob(instance, 1, model="Average")
                        fileheur.write("sbatch %s \n" % (jobname))
                        jobname = CreateHybridSearchJob(instance, nrback, nrforward, setting, model="YFix")
                        fileheur.write("sbatch %s \n" % (jobname))
                        #jobname = CreateMIPJob(instance, "6400b", model="YFix")
                        #fileheur.write("sbatch %s \n" % (jobname))
                        jobname = CreateMIPJob(instance, "all2", model="YFix")
                        fileheur.write("sbatch %s \n" % (jobname))



    if sys.argv[1] == "SDDPScenario":
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
        #!/bin/bash -l
        #
        """)

        nrforward = 1
        sddpnrbackset = ["all5", "all10", "all20", "all50", "all100"]
        for instance in InstanceSet:
            for nrback in sddpnrbackset:
                for setting in ["SingleCut"]:
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting, model="HeuristicYFix")
                    filesddp.write("sbatch %s \n" % (jobname))




    if sys.argv[1] == "SDDP":
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
        #!/bin/bash -l
        #
        """)

        nrforward = 1
        nrback = "all20"
        for instance in InstanceSet:
                for setting in ["Default", "NoEVPI", "NoStrongCut", "SingleCut", "MC" ]:
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting, model="HeuristicYFix", nrtest=0)
                    filesddp.write("sbatch %s \n" % (jobname))

                jobname = CreateSDDPJob(instance, nrback, nrforward, "JustYFix", model="YFix", nrtest=0)
                filesddp.write("sbatch %s \n" % (jobname))

    if sys.argv[1] == "SDDPJYF":
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
               #!/bin/bash -l
               #
               """)

        nrforward = 1
        nrback = "all20"
        for instance in InstanceSet:
            #jobname = CreateSDDPJob(instance, nrback, nrforward, "NoEnhancement", model="YFix", nrtest=0)
            jobname = CreateSDDPJob(instance, nrback, nrforward, "JustYFix", model="YFix", nrtest=0)
            filesddp.write("sbatch %s \n" % (jobname))


        filecompname = "runcompselect.sh"
        filecomp = open(filecompname, 'w')
        filecomp.write("""
        #!/bin/bash -l
        #
        """)
        for instance in InstanceSet:
            jobname = CreateMIPJob(instance, 100, model="YQFix", evaluationpolicy="Fix")
            filecomp.write("sbatch %s \n" % (jobname))
            jobname = CreateMIPJob(instance, 1, model="Average", evaluationpolicy="Fix")
            filecomp.write("sbatch %s \n" % (jobname))
