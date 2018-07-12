#!/usr/bin/python
# script de lancement pour les fichiers
#!/usr/bin/python
# script de lancement pour les fichiers
import sys
import csv
from Constants import Constants

NrScenarioEvaluation = "5000"
ForCIRRELT = True

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
#$ -o /home/thesim/log/outputjobevaluate%s%s%s.txt
ulimit -v 30000000""")

def CreatHeaderQuebec(file):
    file.write("""
#!/bin/bash
#PBS -A abc-123-aa
#PBS -l walltime=30:00:00
#PBS -l nodes=1:ppn=1
#PBS -r n
ulimit -v 16000000
mkdir /tmp/thesim
cd /home/thesim/stochasticmrp/""")


def CreateSDDPJob(instance, nrback, nrforward):
    qsub_filename = "./Jobs/job_sddp_%s_%s_%s" % (instance, nrback, nrforward)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
python scm.py  Solve %s YFix %s RQMC -n 5000 -p Fix -m SDDP --mipsetting Default --nrforward %s
""" % (instance, nrback, nrforward, instance, nrback, nrforward))
    return qsub_filename

def CreateMIPJob(instance):
    qsub_filename = "./Jobs/job_mip_%s" % (instance)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file)
    qsub_file.write("""
python scm.py  Solve %s YFix 6400b RQMC -n 5000 -p Fix -m MIP 
""" % (instance, instance))
    return qsub_filename


if __name__ == "__main__":
    csvfile = open("./Instances/InstancesToSolve.csv", 'rb')
    data_reader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
    instancenameslist = []
    for row in data_reader:
       instancenameslist.append(row)
    InstanceSet = instancenameslist[0]
    instancetosolvename = ""

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
            for nrback in [10, 25, 50, 100, 200]:
                for nrforward in [1, 2, 5, 10, 20, 50]:
                    jobname = CreateSDDPJob(instance, nrback, nrforward)
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
            jobname = CreateMIPJob(instance)
            filemip.write("qsub %s \n" % (jobname) )
