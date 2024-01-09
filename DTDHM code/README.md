# **DTDHM**

DTDHM: Detection of Tandem Duplications Based on Hybrid Methods Using Next-Generation Sequencing Data 

#### 1.Installation

##### 1.1 **Basic requirements:**

- Software: Python, R, SAMtools, BWA, sambamba
- Operating System: Linux
- Python version: 3.8.5 and the higer version
- R version: 4.0.4 and the higer version

##### 1.2 **Required python packages:**

- numpy
- pysam
- sys
- pandas
- pyod
- matplotlib
- numba
- sklearn
- subprocess
- time
- os

##### 1.3 Required R packages:

- DNAcopy
- DNAcopy needs to add the file path to DTDHM.py

#### 2.Running software

##### 2.1 Preprocessing of input files

Usually, the following documents are required:

- A genome reference sequence fasta file. The fasta file must be indexed. You can do the following: $samtools faidx reference.fa
- A bam file from a sample. 
  The bam file needs to be duplicated. You can do the following: $sambamba markdup -r example.bam example_markdup.bam
- Another bam file: Extract inconsistent read pairs from the bam file. You can do the following: $samtools view -b -F 1294 example.bam > example_discordants.bam

The bam file must be sorted and indexed. You can do the following: $samtools sort example.bam; $samtools index example.bam

##### 2.2 Operating command

**python DTDHM.py [reference file] [bam file] [discordant bam] [str1]**

- reference: The path to the fasta file of the genome reference sequence used by the user.
- bam file: The path to the bam file representing the sample used by the user.
- discordant bam file: The path to the bam file containing inconsistent read pairs.

- str1:Length of read. The usual value is 100M.
