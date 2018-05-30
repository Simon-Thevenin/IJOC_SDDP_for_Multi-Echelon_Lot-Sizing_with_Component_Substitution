#!/usr/bin/python
# script de lancement pour les fichiers
#!/usr/bin/python
# script de lancement pour les fichiers
import sys
import csv
from Constants import Constants

NrScenarioEvaluation = "5000"

def CreateSDDPJob(instance, setting):
    qsub_filename = "./Jobs/job_sddp_%s_%s" % (instance, setting)
    qsub_file = open(qsub_filename, 'w')
    qsub_file.write("""
#!/bin/bash -l
#
#$ -cwd
#$ -q idra
#$ -j y
#$ -o /home/thesim/log/outputjobevaluate%s%s.txt
ulimit -v 30000000
python scm.py  Solve %s YFix 10 RQMC -n 10 -p Fix -m SDDP --mipsetting %s
""" % (instance, setting, instance, setting))
    return qsub_filename

def CreateMIPJob(instance):
    qsub_filename = "./Jobs/job_mip_%s" % (instance)
    qsub_file = open(qsub_filename, 'w')
    qsub_file.write("""
#!/bin/bash -l
#
#$ -cwd
#$ -q idra
#$ -j y
#$ -o /home/thesim/log/outputjobevaluate%s.txt
ulimit -v 30000000
python scm.py  Solve %s YFix 10 RQMC -n 10000 -p Fix -m MIP 
""" % (instance, instance))
    return qsub_filename


if __name__ == "__main__":
    csvfile = open("./Instances/InstanceToSolve.csv", 'rb')
    data_reader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
    instancenameslist = []
    for row in data_reader:
       instancenameslist.append(row)
    InstanceSet = instancenameslist[0]
    instancetosolvename = ""

    if sys.argv[1] == "SDDP":
        settings = ["Default", "NoOneTree", "NoFirstCuts", "NoValidInequalities", "NoStongCut"]
        # Create the sh file for resolution
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
#!/bin/bash -l
#
""")

        for instance in InstanceSet:
            for setting in settings:
                jobname = CreateSDDPJob(instance, setting)
                filesddp.write("qsub %s \n" % (jobname) )


    if sys.argv[1] == "MIP":
       # Create the sh file for resolution
        filemipname = "runallsddp.sh"
        filemip = open(filemipname, 'w')
        filemip.write("""
#!/bin/bash -l
#
""")

        for instance in InstanceSet:
            jobname = CreateMIPJob(instance)
            filemip.write("qsub %s \n" % (jobname) )
