#!/usr/bin/python
# script de lancement pour les fichiers
#!/usr/bin/python
# script de lancement pour les fichiers
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
#SBATCH --partition=MPI-short
#
# Nombre de noeuds
#SBATCH --nodes=1
# Nombre de processus MPI par noeud
#SBATCH --ntasks-per-node=1
#
# Temps de presence du job
#SBATCH --time=06:00:00
#
# Adresse mel de l'utilisateur
#
# Envoi des mails
#SBATCH --mail-type=abort,end
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

def CreateSDDPJob(instance, nrback, nrforward, setting, model = "YFix"):
    qsub_filename = "./Jobs/job_sddp_%s_%s_%s_%s_%s" % (instance, nrback, nrforward, setting, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
srun python scm.py  Solve %s %s %s RQMC -n 5000 -p Re-solve -m SDDP --mipsetting %s --nrforward %s  > /home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt 
""" % (instance, model, nrback, setting, nrforward))
    return qsub_filename

def CreateMLLocalSearchJob(instance, nrback, nrforward, setting, model = "YFix"):
    qsub_filename = "./Jobs/job_mllocalsearch_%s_%s_%s_%s_%s" % (instance, nrback, nrforward, setting, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
srun python scm.py  Solve %s %s %s RQMC -n 5000 -p Re-solve -m MLLocalSearch --mipsetting %s --nrforward %s >/home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt
""" % (instance, model, nrback, setting, nrforward))
    return qsub_filename

def CreateHybridSearchJob(instance, nrback, nrforward, setting, model = "YFix"):
    qsub_filename = "./Jobs/job_hybrid_%s_%s_%s_%s_%s" % (instance, nrback, nrforward, setting, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
srun python scm.py  Solve %s %s %s RQMC -n 5000 -p Re-solve -m Hybrid --mipsetting %s --nrforward %s >/home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt
""" % (instance, model, nrback, setting, nrforward))
    return qsub_filename

def CreateMIPJob(instance, scenariotree, model = "YFix"):
    qsub_filename = "./Jobs/job_mip_%s_%s_%s" % (instance, scenariotree, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file)
    qsub_file.write("""
srun python scm.py  Solve %s %s %s RQMC -n 5000 -p Re-solve -m MIP  >/home/LS2N/thevenin-s/log/output-${SLURM_JOB_ID}.txt 
""" % (instance, model, scenariotree))
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
    instancetosolvename = ""

    scenariotreeset = ["6400b"] #["all2", "all5"]#, "allDIX", "all20"]
    #sddpnrbackset = ["allDIX", "all20", "50-50-10", "all50" ]#,10,20] #[2, 5], 10, 20]
    sddpnrbackset = ["all50"]
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
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting, model="YFix")
                    fileheur.write("sbatch %s \n" % (jobname))
                   # jobname = CreateMLLocalSearchJob(instance, nrback, nrforward, setting, model="YFix")
                   # fileheur.write("sbatch %s \n" % (jobname))
                    jobname = CreateHybridSearchJob(instance, nrback, nrforward, setting, model="YFix")
                    fileheur.write("sbatch %s \n" % (jobname))



            jobname = CreateMIPJob(instance, 100, model="YQFix")
            fileheur.write("sbatch %s \n" % (jobname))

            # for scenariotree in scenariotreeset:
            #          jobname = CreateMIPJob(instance, scenariotree, model="HeuristicYFix")
            #          fileheur.write("sbatch %s \n" % (jobname))
             #   jobname = CreatePHJob(instance, scenariotree, model = "HeuristicYFix")
             #   fileheur.write("sbatch %s \n" % (jobname))



    if sys.argv[1] == "SDDP":
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
        #!/bin/bash -l
        #
        """)

        InstanceSet = ["01_Lumpy_b2_fe25_en_rk50_ll0_l20_HFalse10_c2",
                       "02_Lumpy_b2_fe25_en_rk50_ll0_l20_HFalse10_c2",
                       "03_Lumpy_b2_fe25_en_rk50_ll0_l20_HFalse10_c2",
                       "04_Lumpy_b2_fe25_en_rk50_ll0_l20_HFalse10_c2",
                       "05_Lumpy_b2_fe25_en_rk50_ll0_l20_HFalse10_c2"]

        for instance in InstanceSet:
            for nrback in sddpnrbackset:
                for setting in ["Default"]:
                    nrforward = 1
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting, model="HeuristicYFix")
                    filesddp.write("sbatch %s \n" % (jobname))



        # for instance in InstanceSet:
        #    for nrback in sddpnrbackset:
        #         for setting in [ "Default", "NoFirstCuts", "NoEVPI", "NoStongCut", "NoSingleTree", "WithLPTree", "WithFixedSetups", "WithFixedSetupsNoScenarioTree" ]:
        #
        #             nrforward = 1
        #             jobname = CreateSDDPJob(instance, nrback, nrforward, setting)
        #             filesddp.write("qsub %s \n" % (jobname) )


    if sys.argv[1] == "MIP":
       # Create the sh file for resolution
        filemipname = "runallmip.sh"
        filemip = open(filemipname, 'w')
        filemip.write("""
#!/bin/bash -l
#
""")

        for instance in InstanceSet:
            for scenariotree in scenariotreeset:
                jobname = CreateMIPJob(instance, scenariotree)
                filemip.write("qsub %s \n" % (jobname) )

    if sys.argv[1] == "PH":
       # Create the sh file for resolution
        filemipname = "runallph.sh"
        filemip = open(filemipname, 'w')
        filemip.write("""
#!/bin/bash -l
#
""")

        for instance in InstanceSet:
            for scenariotree in scenariotreeset:
                jobname = CreatePHJob(instance, scenariotree)
                filemip.write("sbatch %s \n" % (jobname) )
