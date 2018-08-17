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
mkdir -p /tmp/thesim
mkdir -p /tmp/thesim/Evaluations
mkdir -p /tmp/thesim/Solutions
mkdir -p /tmp/thesim/CPLEXLog
""")

def CreatHeaderQuebec(file):
    file.write("""
#!/bin/bash
#PBS -A abc-123-aa
#PBS -l walltime=30:00:00
#PBS -l nodes=1:ppn=1
#PBS -r n
mkdir /tmp/thesim
cd /home/thesim/stochasticmrp/
""")


def CreateSDDPJob(instance, nrback, nrforward, setting):
    qsub_filename = "./Jobs/job_sddp_%s_%s_%s_%s" % (instance, nrback, nrforward, setting)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file )
    qsub_file.write("""
#$ -o /home/thesim/log/outputjobevaluate%s%s%s%s.txt
ulimit -v 16000000
python scm.py  Solve %s YFix %s RQMC -n 5000 -p Fix -m SDDP --mipsetting %s --nrforward %s
""" % (instance, nrback, setting, nrforward, instance, nrback, setting, nrforward))
    return qsub_filename

def CreateMIPJob(instance, scenariotree):
    qsub_filename = "./Jobs/job_mip_%s_%s" % (instance, scenariotree)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file)
    qsub_file.write("""
#$ -o /home/thesim/log/outputjobevaluate%s%s.txt
ulimit -v 16000000
python scm.py  Solve %s YFix %s RQMC -n 5000 -p Fix -m MIP 
""" % (instance, scenariotree, instance, scenariotree))
    return qsub_filename


def CreatePHJob(instance, scenariotree):
    qsub_filename = "./Jobs/job_ph_%s_%s" % (instance, scenariotree)
    qsub_file = open(qsub_filename, 'w')
    CreatHeader(qsub_file)
    qsub_file.write("""
#$ -o /home/thesim/log/outputjobevaluateph%s%s.txt
ulimit -v 16000000
python scm.py  Solve %s YFix %s RQMC -n 5000 -p Fix -m PH 
""" % (instance, scenariotree, instance, scenariotree))
    return qsub_filename

if __name__ == "__main__":
    csvfile = open("./Instances/InstancesToSolve.csv", 'rb')
    data_reader = csv.reader(csvfile, delimiter=",", skipinitialspace=True)
    instancenameslist = []
    for row in data_reader:
       instancenameslist.append(row)
    InstanceSet = instancenameslist[0]
    instancetosolvename = ""

    scenariotreeset = ["all2", "all5", "allDIX"]
    sddpnrbackset = [2, 5, 10]



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
                for setting in [ "Default", "NoFirstCuts", "NoEVPI", "NoStongCut", "NoSingleTree" ]:

                    nrforward = 1
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting)
                    filesddp.write("qsub %s \n" % (jobname) )

    if sys.argv[1] == "NST":
        # settings = ["Default", "NoFirstCuts", "NoEVPI", "NoStongCut"]
        # Create the sh file for resolution
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
    #!/bin/bash -l
    #
    """)

        for instance in InstanceSet:
            for nrback in sddpnrbackset:
                for setting in [ "NoSingleTree"]:
                    nrforward = 1
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting)
                    filesddp.write("qsub %s \n" % (jobname))

    if sys.argv[1] == "0102":
        # settings = ["Default", "NoFirstCuts", "NoEVPI", "NoStongCut"]
        # Create the sh file for resolution
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
      #!/bin/bash -l
      #
      """)

        for instance in ["01_NonStationary_b2_fe25_en_rk50_ll0_l20_HFalse_c2", "02_NonStationary_b2_fe25_en_rk50_ll0_l20_HFalse_c2"]:
            for nrback in sddpnrbackset:
                for setting in [ "Default", "NoFirstCuts", "NoEVPI", "NoStongCut", "NoSingleTree" ]:
                    nrforward = 1
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting)
                    filesddp.write("qsub %s \n" % (jobname))

    if sys.argv[1] == "LT":
        # settings = ["Default", "NoFirstCuts", "NoEVPI", "NoStongCut"]
        # Create the sh file for resolution
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
        #!/bin/bash -l
        #
        """)

        for instance in ["G0041111_NonStationary_b2_fe25_en_rk50_ll0_l20_HTrue6_c2",
                          "K0017311_NonStationary_b2_fe25_en_rk50_ll0_l20_HTrue6_c2"]:
            for nrback in [5]:
                for setting in ["Default"]:
                    nrforward = 1
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting)
                    filesddp.write("qsub %s \n" % (jobname))

        filemipname = "runallmip.sh"
        filemip = open(filemipname, 'w')
        filemip.write("""
        #!/bin/bash -l
        #
        """)

        for instance in ["G0041111_NonStationary_b2_fe25_en_rk50_ll0_l20_HTrue6_c2",
                          "K0017311_NonStationary_b2_fe25_en_rk50_ll0_l20_HTrue6_c2"]:
            for scenariotree in ["all5"]:
                jobname = CreateMIPJob(instance, scenariotree)
                filemip.write("qsub %s \n" % (jobname))

    if sys.argv[1] == "15":
        # settings = ["Default", "NoFirstCuts", "NoEVPI", "NoStongCut"]
        # Create the sh file for resolution
        filesddpname = "runallsddp.sh"
        filesddp = open(filesddpname, 'w')
        filesddp.write("""
           #!/bin/bash -l
           #
           """)

        for instance in ["K0011111_NonStationary_b2_fe25_en_rk50_ll0_l20_HFalse_c2"]:
            for nrback in [15]:
                for setting in ["Default"]:
                    nrforward = 1
                    jobname = CreateSDDPJob(instance, nrback, nrforward, setting)
                    filesddp.write("qsub %s \n" % (jobname))

        filemipname = "runallmip.sh"
        filemip = open(filemipname, 'w')
        filemip.write("""
           #!/bin/bash -l
           #
           """)

        for instance in ["K0011111_NonStationary_b2_fe25_en_rk50_ll0_l20_HFalse_c2"]:
            for scenariotree in ["all15"]:
                jobname = CreateMIPJob(instance, scenariotree)
                filemip.write("qsub %s \n" % (jobname))

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
                filemip.write("qsub %s \n" % (jobname) )
