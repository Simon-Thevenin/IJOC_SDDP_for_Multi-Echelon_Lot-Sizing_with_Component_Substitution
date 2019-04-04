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
    if ForCIRRELT:
        CreatHeaderCirrelt(file)
    else:
        CreatHeaderQuebec(file)

def CreatHeaderCirrelt(file):
    file.write("""
#!/bin/bash -l
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
    file.write("""
#!/bin/bash
#SBATCH --time=00:01:00

mkdir /tmp/thesim

module load python/2.7
module load scipy-stack
module load python/2.7
module load cplex/2.9
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --upgrade pip
cd /home/thesim/installcplex/CPLEX_Studio129/cplex/python/2.7/x86-64_linux/
python setup.py install
cd /home/thesim/
pip install openpyxl --upgrade --pre
pip install networkx
pip freeze > requirements.txt


cd /home/thesim/ProjectJFY/scm

""")


def CreateSDDPJob(instance, nrback, nrforward, setting, model = "YFix"):
    qsub_filename = "./Jobs/job_sddp_%s_%s_%s_%s_%s" % (instance, nrback, nrforward, setting, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
#$ -o /home/thesim/log/outputjobevaluate%s%s%s%s%s.txt
ulimit -v 16000000
python scm.py  Solve %s %s %s RQMC -n 5000 -p Fix -m SDDP --mipsetting %s --nrforward %s
""" % (instance, model, nrback, setting, nrforward, instance, model, nrback, setting, nrforward))
    return qsub_filename

def CreateMLLocalSearchJob(instance, nrback, nrforward, setting, model = "YFix"):
    qsub_filename = "./Jobs/job_mllocalsearch_%s_%s_%s_%s_%s" % (instance, nrback, nrforward, setting, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
#$ -o /home/thesim/log/outputjobevaluate%s%s%s%s%s.txt
ulimit -v 16000000
python scm.py  Solve %s %s %s RQMC -n 5000 -p Fix -m MLLocalSearch --mipsetting %s --nrforward %s
""" % (instance, model, nrback, setting, nrforward, instance, model, nrback, setting, nrforward))
    return qsub_filename


def CreateMIPJob(instance, scenariotree, model = "YFix"):
    qsub_filename = "./Jobs/job_mip_%s_%s_%s" % (instance, scenariotree, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file)
    qsub_file.write("""
#$ -o /home/thesim/log/outputjobevaluate%s%s%s.txt
ulimit -v 16000000
python scm.py  Solve %s %s %s RQMC -n 5000 -p Fix -m MIP 
""" % (instance, model, scenariotree, instance, model, scenariotree))
    return qsub_filename


def CreatePHJob(instance, scenariotree, model = "YFix"):
    qsub_filename = "./Jobs/job_ph_%s_%s_%s" % (instance, scenariotree, model)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file)
    qsub_file.write("""
#$ -o /home/thesim/log/outputjobevaluateph%s%s%s.txt
ulimit -v 16000000
python scm.py  Solve %s %s %s RQMC -n 5000 -p Fix -m PH 
""" % (instance, model, scenariotree, instance, model, scenariotree))
    return qsub_filename

if __name__ == "__main__":
    csvfile = open("./Instances/InstancesToSolve.csv", 'rb')
    data_reader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
    instancenameslist = []
    for row in data_reader:
       instancenameslist.append(row)
    InstanceSet = instancenameslist[0]
    instancetosolvename = ""

    scenariotreeset = ["all2", "all5"]#, "allDIX", "all20"]
    sddpnrbackset = [2, 5]#, 10, 20]

    if sys.argv[1] == "H":
        fileheurname = "runallheur.sh"
        fileheur = open(fileheurname, 'w')
        fileheur.write("""
        #!/bin/bash -l
        #
        """)

        InstanceSet = ["SuperSmallIntance"]

        for instance in InstanceSet:
            for nrback in sddpnrbackset:
                for setting in ["Default"]:
                    nrforward = 1
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting, model = "YFix")
                    fileheur.write("sbatch %s \n" % (jobname))
                    jobname = CreateMLLocalSearchJob(instance, nrback, nrforward, setting, model="YFix")
                    fileheur.write("sbatch %s \n" % (jobname))


            jobname = CreateMIPJob(instance, 100, model="YQFix")
            fileheur.write("sbatch %s \n" % (jobname))

            for scenariotree in scenariotreeset:
                jobname = CreateMIPJob(instance, scenariotree, model = "HeuristicYFix")
                fileheur.write("sbatch %s \n" % (jobname))
                jobname = CreatePHJob(instance, scenariotree, model = "HeuristicYFix")
                fileheur.write("sbatch %s \n" % (jobname))



    if sys.argv[1] == "SDDP":
        #settings = ["Default", "NoFirstCuts", "NoEVPI", "NoStongCut"]
        # Create the sh file for resolution
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
#!/bin/bash -l
#
""")

        for instance in InstanceSet:
           for nrback in sddpnrbackset:
                for setting in [ "Default", "NoFirstCuts", "NoEVPI", "NoStongCut", "NoSingleTree", "WithLPTree", "WithFixedSetups", "WithFixedSetupsNoScenarioTree" ]:

                    nrforward = 1
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting)
                    filesddp.write("qsub %s \n" % (jobname) )


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
